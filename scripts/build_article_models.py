from __future__ import annotations

import io
import json
import re
import unicodedata
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"
REPORT_DIR = ROOT / "results"

for folder in (DATA_DIR, FIG_DIR, TABLE_DIR):
    folder.mkdir(parents=True, exist_ok=True)


SIDRA_URL = (
    "https://apisidra.ibge.gov.br/values/t/74/n6/in%20n3%2041/"
    "v/106/p/all/c80/2682?formato=json"
)


def parse_sidra_value(value: object) -> float:
    if value is None:
        return np.nan
    text = str(value).strip()
    if text in {"", "-", "..", "...", "X"}:
        return np.nan if text != "-" else 0.0
    text = text.replace(".", "").replace(",", ".")
    return float(text)


def normalize_name(value: object) -> str:
    text = str(value).strip().upper()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    return text.strip("_")


def download_sidra_leite() -> pd.DataFrame:
    cache = DATA_DIR / "sidra_leite_pr_raw.json"
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
    else:
        response = requests.get(SIDRA_URL, timeout=120)
        response.raise_for_status()
        raw = response.json()
        cache.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    rows = raw[1:]
    df = pd.DataFrame(rows)
    df = df.rename(
        columns={
            "D1C": "codigo_municipio",
            "D1N": "municipio",
            "D3C": "ano",
            "V": "producao_leite_mil_litros",
        }
    )
    df = df[["codigo_municipio", "municipio", "ano", "producao_leite_mil_litros"]]
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
    df["producao_leite_mil_litros"] = df["producao_leite_mil_litros"].map(parse_sidra_value)
    df["municipio"] = df["municipio"].str.replace(r"\s+-\s+PR$", "", regex=True)
    df = df.dropna(subset=["ano", "producao_leite_mil_litros"])
    return df.sort_values(["codigo_municipio", "ano"]).reset_index(drop=True)


def read_inmet_csv_from_zip(year: int) -> pd.DataFrame | None:
    out = DATA_DIR / f"inmet_castro_a819_{year}.csv"
    if out.exists():
        cached = pd.read_csv(out)
        normalized_cached = {normalize_name(c) for c in cached.columns}
        if "Data" in cached.columns or "DATA" in normalized_cached or "DATA_YYYY_MM_DD" in normalized_cached:
            return cached
        print(f"INMET {year}: cache inválido encontrado; baixando novamente.")

    url = f"https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip"
    response = requests.get(url, timeout=180)
    if response.status_code != 200:
        print(f"INMET {year}: download indisponível ({response.status_code})")
        return None

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        candidates = [
            name
            for name in zf.namelist()
            if name.lower().endswith(".csv")
            and ("a819" in name.lower() or "castro" in name.lower())
        ]
        if not candidates:
            print(f"INMET {year}: estação A819/CASTRO não encontrada no ZIP")
            return None
        name = candidates[0]
        data = zf.read(name)

    text = None
    for enc in ("utf-8-sig", "cp1252", "latin1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise UnicodeDecodeError("inmet", data, 0, 1, "Codificação não identificada")

    lines = text.splitlines()
    header_idx = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("Data;") or line.startswith("DATA (") or line.startswith("DATA;")
    )
    csv_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(
        io.StringIO(csv_text),
        sep=";",
        decimal=",",
        na_values=["", "null", "NULL", "9999", "-9999"],
    )
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    df.to_csv(out, index=False)
    return df


def build_climate_features(start_year: int = 2006, end_year: int = 2019) -> pd.DataFrame:
    cache = DATA_DIR / "clima_castro_a819_anual.csv"
    if cache.exists():
        return pd.read_csv(cache)

    frames = []
    for year in range(start_year, end_year + 1):
        print(f"Processando INMET {year}...")
        df = read_inmet_csv_from_zip(year)
        if df is None or df.empty:
            continue

        normalized = {normalize_name(c): c for c in df.columns}
        wanted = {
            "data": ["DATA", "DATA_YYYY_MM_DD"],
            "precipitacao_mm": ["PRECIPITACAO_TOTAL_HORARIO_MM"],
            "temperatura_c": ["TEMPERATURA_DO_AR_BULBO_SECO_HORARIA_C"],
            "radiacao_kj_m2": ["RADIACAO_GLOBAL_KJ_M2"],
            "umidade_pct": ["UMIDADE_RELATIVA_DO_AR_HORARIA"],
            "vento_ms": ["VENTO_VELOCIDADE_HORARIA_M_S"],
        }

        selected = {}
        for target, options in wanted.items():
            for option in options:
                if option in normalized:
                    selected[target] = normalized[option]
                    break
        if "data" not in selected:
            raise ValueError(f"Coluna de data não encontrada em {year}: {list(df.columns)[:5]}")

        value_cols = {v: k for k, v in selected.items() if k != "data"}
        small = df[[selected["data"], *value_cols.keys()]].rename(
            columns={selected["data"]: "Data", **value_cols}
        ).copy()
        small["ano"] = pd.to_datetime(small["Data"], errors="coerce").dt.year
        for col in value_cols.values():
            small[col] = pd.to_numeric(small[col], errors="coerce")
        frames.append(small)

    climate_hourly = pd.concat(frames, ignore_index=True)
    grouped = climate_hourly.groupby("ano", as_index=False).agg(
        precipitacao_anual_mm=("precipitacao_mm", "sum"),
        temperatura_media_c=("temperatura_c", "mean"),
        temperatura_max_media_c=("temperatura_c", "max"),
        radiacao_media_kj_m2=("radiacao_kj_m2", "mean"),
        umidade_media_pct=("umidade_pct", "mean"),
        vento_medio_ms=("vento_ms", "mean"),
        horas_validas_clima=("temperatura_c", "count"),
    )
    grouped["dias_secos_aprox"] = (
        climate_hourly.assign(chuva=lambda d: d["precipitacao_mm"].fillna(0))
        .groupby(["ano", pd.to_datetime(climate_hourly["Data"], errors="coerce").dt.date])["chuva"]
        .sum()
        .reset_index()
        .assign(seco=lambda d: d["chuva"] < 1.0)
        .groupby("ano")["seco"]
        .sum()
        .reindex(grouped["ano"])
        .to_numpy()
    )
    grouped.to_csv(cache, index=False)
    return grouped


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["codigo_municipio", "ano"]).copy()
    group = df.groupby("codigo_municipio")["producao_leite_mil_litros"]
    df["leite_lag_1"] = group.shift(1)
    df["leite_lag_2"] = group.shift(2)
    df["leite_media_3anos"] = group.shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["crescimento_lag_1_pct"] = (df["leite_lag_1"] - df["leite_lag_2"]) / df["leite_lag_2"]
    return df


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape}


def train_models(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline]]:
    model_df = base.dropna(subset=["leite_lag_1", "leite_lag_2", "leite_media_3anos"]).copy()
    model_df = model_df[(model_df["ano"] >= 2008) & (model_df["ano"] <= 2019)]

    feature_cols = [
        "codigo_municipio",
        "municipio",
        "ano",
        "leite_lag_1",
        "leite_lag_2",
        "leite_media_3anos",
        "crescimento_lag_1_pct",
        "precipitacao_anual_mm",
        "temperatura_media_c",
        "temperatura_max_media_c",
        "radiacao_media_kj_m2",
        "umidade_media_pct",
        "vento_medio_ms",
        "dias_secos_aprox",
    ]
    target = "producao_leite_mil_litros"

    train = model_df[model_df["ano"] <= 2014]
    valid = model_df[(model_df["ano"] >= 2015) & (model_df["ano"] <= 2016)]
    test = model_df[model_df["ano"] >= 2017]
    train_tune = pd.concat([train, valid], ignore_index=True)

    categorical = ["codigo_municipio", "municipio"]
    numeric = [c for c in feature_cols if c not in categorical]
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ]
    )

    rf = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", RandomForestRegressor(random_state=42, n_jobs=1)),
        ]
    )
    xgb = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=1,
                    tree_method="hist",
                ),
            ),
        ]
    )

    # Each fold validates on a complete future year, across all municipalities.
    years = train_tune["ano"].to_numpy()
    cv = [
        (np.flatnonzero(years < validation_year), np.flatnonzero(years == validation_year))
        for validation_year in (2015, 2016)
    ]
    cv = [(train_idx, valid_idx) for train_idx, valid_idx in cv if len(train_idx) and len(valid_idx)]
    searches = {
        "Random Forest": GridSearchCV(
            TransformedTargetRegressor(regressor=rf, func=np.log1p, inverse_func=np.expm1),
            {
                "regressor__model__n_estimators": [100, 300],
                "regressor__model__max_depth": [8, None],
                "regressor__model__min_samples_leaf": [2],
            },
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
        ),
        "XGBoost": GridSearchCV(
            TransformedTargetRegressor(regressor=xgb, func=np.log1p, inverse_func=np.expm1),
            {
                "regressor__model__n_estimators": [100, 300],
                "regressor__model__max_depth": [2, 4],
                "regressor__model__learning_rate": [0.05],
                "regressor__model__subsample": [0.8],
                "regressor__model__colsample_bytree": [0.8],
            },
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
        ),
    }

    X_train = train_tune[feature_cols]
    y_train = train_tune[target]
    X_test = test[feature_cols]
    y_test = test[target].to_numpy()

    all_metrics = []
    pred_frames = []
    fitted = {}

    naive_pred = test["leite_lag_1"].to_numpy()
    naive_metrics = metrics(y_test, naive_pred)
    naive_metrics.update({"modelo": "Persistencia lag-1", "melhores_parametros": "-"})
    all_metrics.append(naive_metrics)

    for name, search in searches.items():
        print(f"Treinando {name}...")
        search.fit(X_train, y_train)
        pred = search.predict(X_test)
        m = metrics(y_test, pred)
        m.update({"modelo": name, "melhores_parametros": json.dumps(search.best_params_, ensure_ascii=False)})
        all_metrics.append(m)
        fitted[name] = search.best_estimator_
        pred_frames.append(
            test[["codigo_municipio", "municipio", "ano", target]].assign(
                modelo=name, predicao_mil_litros=pred, erro_mil_litros=lambda d: d["predicao_mil_litros"] - d[target]
            )
        )

    pred_frames.append(
        test[["codigo_municipio", "municipio", "ano", target]].assign(
            modelo="Persistencia lag-1",
            predicao_mil_litros=naive_pred,
            erro_mil_litros=lambda d: d["predicao_mil_litros"] - d[target],
        )
    )
    metrics_df = pd.DataFrame(all_metrics)[["modelo", "MAE", "RMSE", "R2", "MAPE_%", "melhores_parametros"]]
    predictions = pd.concat(pred_frames, ignore_index=True)
    return metrics_df, predictions, fitted


def make_figures(metrics_df: pd.DataFrame, predictions: pd.DataFrame, fitted: dict[str, Pipeline]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    order = metrics_df.sort_values("RMSE")["modelo"]
    ax.barh(order, metrics_df.set_index("modelo").loc[order, "RMSE"], color=["#2a9d8f", "#457b9d", "#e76f51"])
    ax.set_title("Comparação de modelos: RMSE no teste (2017-2019)")
    ax.set_xlabel("RMSE (mil litros)")
    ax.set_ylabel("Modelo")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figura_01_rmse_modelos.png", dpi=220)
    plt.close(fig)

    best_model = metrics_df[~metrics_df["modelo"].str.contains("Persistencia")].sort_values("RMSE").iloc[0]["modelo"]
    best_pred = predictions[predictions["modelo"] == best_model]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(best_pred["producao_leite_mil_litros"], best_pred["predicao_mil_litros"], s=18, alpha=0.55, color="#264653")
    min_v = min(best_pred["producao_leite_mil_litros"].min(), best_pred["predicao_mil_litros"].min())
    max_v = max(best_pred["producao_leite_mil_litros"].max(), best_pred["predicao_mil_litros"].max())
    ax.plot([min_v, max_v], [min_v, max_v], color="#e76f51", linewidth=2)
    ax.set_title(f"Produção observada e prevista - {best_model}")
    ax.set_xlabel("Produção observada (mil litros)")
    ax.set_ylabel("Produção prevista (mil litros)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figura_02_observado_predicho.png", dpi=220)
    plt.close(fig)

    top_mun = (
        best_pred.groupby("municipio")["producao_leite_mil_litros"]
        .mean()
        .sort_values(ascending=False)
        .head(4)
        .index
    )
    plot_df = best_pred[best_pred["municipio"].isin(top_mun)]
    fig, ax = plt.subplots(figsize=(10, 6))
    for municipio, group in plot_df.groupby("municipio"):
        ax.plot(group["ano"], group["producao_leite_mil_litros"], marker="o", linewidth=1.8, label=f"{municipio} obs.")
        ax.plot(group["ano"], group["predicao_mil_litros"], marker="x", linestyle="--", linewidth=1.4, alpha=0.8)
    ax.set_title(f"Produção observada e prevista, 2017-2019 - {best_model}")
    ax.set_xlabel("Año")
    ax.set_ylabel("Producción de leche (mil litros)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figura_03_series_municipios_lideres.png", dpi=220)
    plt.close(fig)

    model = fitted[best_model]
    if isinstance(model, TransformedTargetRegressor):
        pipeline = model.regressor_
    else:
        pipeline = model
    reg = pipeline.named_steps["model"]
    pre = pipeline.named_steps["preprocess"]
    names = pre.get_feature_names_out()
    if hasattr(reg, "feature_importances_"):
        importances = pd.DataFrame({"variable": names, "importancia": reg.feature_importances_})
        importances["variable"] = importances["variable"].str.replace("cat__", "", regex=False).str.replace("num__", "", regex=False)
        labels = {
            "leite_lag_1": "Produção no ano anterior",
            "leite_lag_2": "Produção há dois anos",
            "leite_media_3anos": "Média dos três anos anteriores",
            "crescimento_lag_1_pct": "Variação percentual defasada",
            "ano": "Ano",
            "precipitacao_anual_mm": "Precipitação anual (mm)",
            "temperatura_media_c": "Temperatura média (°C)",
            "temperatura_max_media_c": "Temperatura máxima anual (°C)",
            "radiacao_media_kj_m2": "Radiação média (kJ/m²)",
            "umidade_media_pct": "Umidade média (%)",
            "vento_medio_ms": "Velocidade média do vento (m/s)",
            "dias_secos_aprox": "Dias secos (aprox.)",
        }
        importances["variable"] = importances["variable"].replace(labels)
        importances = importances.sort_values("importancia", ascending=False).head(15)
        importances.to_csv(TABLE_DIR / "importancia_variables.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(importances["variable"][::-1], importances["importancia"][::-1], color="#2a9d8f")
        ax.set_title(f"Importância dos atributos - {best_model}")
        ax.set_xlabel("Importância relativa")
        ax.set_ylabel("Atributo")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "figura_04_importancia_variables.png", dpi=220)
        plt.close(fig)


def make_workflow_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        "IBGE / PPM\nProdução municipal\n2006-2019",
        "INMET A819\nClima anual\nCastro/PR",
        "Painel município-ano\nDefasagens + variáveis\nagroclimáticas",
        "Ajuste e seleção\nValidação temporal\n2015-2016",
        "Teste temporal\n2017-2019\nPersistência, RF, XGBoost",
    ]
    ys = [4.35, 3.55, 2.75, 1.95, 1.15]
    for y, label in zip(ys, boxes):
        ax.text(4, y, label, ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round,pad=0.45", "facecolor": "#edf4f4", "edgecolor": "#287271", "linewidth": 1.2})
    for y in (3.98, 3.18, 2.38, 1.58):
        ax.annotate("", xy=(4, y - 0.25), xytext=(4, y + 0.15),
                    arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.4})
    ax.text(4, 0.42, "Métricas: MAE · RMSE · R² · MAPE", ha="center", va="center", fontsize=9, color="#333333")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figura_00_fluxo_metodologico.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(metrics_df: pd.DataFrame, base: pd.DataFrame) -> None:
    best = metrics_df[~metrics_df["modelo"].str.contains("Persistencia")].sort_values("RMSE").iloc[0]
    baseline = metrics_df[metrics_df["modelo"].str.contains("Persistencia")].iloc[0]
    report = f"""# Resultados para artículo

## Tema

**Predicción de la producción lechera municipal en Paraná mediante aprendizaje automático y datos agroclimáticos públicos**

## Base integrada

- Fuente agropecuaria: SIDRA/IBGE, Producción de la Pecuaria Municipal, leche, municipios de Paraná.
- Fuente climática: INMET, estación automática A819 - Castro/PR, históricos horarios agregados por año.
- Unidad analítica: municipio-año.
- Registros en la base integrada: {len(base):,}.
- Periodo usado en modelado: 2008-2019, por disponibilidad de rezagos y clima.
- Prueba temporal: 2017-2019.

## Variables principales

- Producción de leche en mil litros.
- Rezagos: producción del año anterior, producción de dos años antes y media móvil de tres años.
- Variables climáticas anuales: precipitación acumulada, temperatura media, temperatura máxima observada, radiación media, humedad media, viento medio y días secos aproximados.
- Identificador municipal como efecto espacial tabular.

## Métricas en prueba

{metrics_df.to_markdown(index=False)}

## Mejor modelo de aprendizaje automático y comparación con baseline

El mejor algoritmo de aprendizaje automático por RMSE fue **{best['modelo']}**, con R² = **{best['R2']:.3f}**, RMSE = **{best['RMSE']:.2f} mil litros** y MAPE = **{best['MAPE_%']:.2f}%**. Sin embargo, la persistencia lag-1 obtuvo RMSE = **{baseline['RMSE']:.2f} mil litros** y superó a RF y XGBoost en este periodo de prueba; por tanto, los algoritmos de ML no demostraron mejora predictiva frente al baseline.

## Archivos generados

- `data/base_integrada_leite_agroclima.csv`
- `results/tables/metricas_modelos.csv`
- `results/tables/predicciones_prueba.csv`
- `results/figures/figura_01_rmse_modelos.png`
- `results/figures/figura_02_observado_predicho.png`
- `results/figures/figura_03_series_municipios_lideres.png`
- `results/figures/figura_04_importancia_variables.png`
- `docs/Artigo_IIIWCAA_Predicao_Leite_Parana.docx`
"""
    (REPORT_DIR / "resumen_resultados_articulo.md").write_text(report, encoding="utf-8")


def main() -> None:
    leite = download_sidra_leite()
    clima = build_climate_features()
    base = leite.merge(clima, on="ano", how="inner")
    base = add_time_features(base)
    base.to_csv(DATA_DIR / "base_integrada_leite_agroclima.csv", index=False)

    metrics_df, predictions, fitted = train_models(base)
    metrics_df.to_csv(TABLE_DIR / "metricas_modelos.csv", index=False)
    predictions.to_csv(TABLE_DIR / "predicciones_prueba.csv", index=False)
    make_workflow_figure()
    make_figures(metrics_df, predictions, fitted)
    write_report(metrics_df, base)
    print(metrics_df)


if __name__ == "__main__":
    main()
