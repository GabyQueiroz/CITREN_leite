from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "data" / "base_integrada_leite_agroclima.csv"
TABLE_DIR = ROOT / "results" / "decline_risk" / "tables"
FIG_DIR = ROOT / "results" / "decline_risk" / "figures"
REPORT_PATH = ROOT / "results" / "decline_risk" / "resultados_alerta_queda.md"
EVENT_THRESHOLD = -0.10
SEED = 42

for directory in (TABLE_DIR, FIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def prepare_panel(source: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = source.sort_values(["codigo_municipio", "ano"]).copy()
    grouped = df.groupby("codigo_municipio", sort=False)
    previous = grouped["producao_leite_mil_litros"].shift(1)
    previous2 = grouped["producao_leite_mil_litros"].shift(2)
    df["variacao_alvo"] = df["producao_leite_mil_litros"] / previous - 1
    df["queda_10_pct"] = (df["variacao_alvo"] <= EVENT_THRESHOLD).astype("Int64")
    df["leite_lag_1"] = previous
    df["leite_lag_2"] = previous2
    df["crescimento_previo"] = previous / previous2 - 1

    climate = [
        "precipitacao_anual_mm",
        "temperatura_media_c",
        "temperatura_max_media_c",
        "radiacao_media_kj_m2",
        "umidade_media_pct",
        "vento_medio_ms",
        "dias_secos_aprox",
    ]
    # Shift annual weather to ensure predictors are available before outcome year.
    for col in climate:
        df[f"{col}_lag_1"] = df.groupby("codigo_municipio", sort=False)[col].shift(1)

    numeric = ["ano", "leite_lag_1", "leite_lag_2", "crescimento_previo"] + [
        f"{col}_lag_1" for col in climate
    ]
    categorical = ["codigo_municipio"]
    df = df.dropna(subset=["variacao_alvo", *numeric]).copy()
    df = df[df["ano"].between(2008, 2019)].copy()
    df["queda_10_pct"] = df["queda_10_pct"].astype(int)
    return df, numeric, categorical


def make_pipeline(estimator, numeric: list[str], categorical: list[str]) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            ("municipio", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline([("prep", prep), ("model", estimator)])


def class_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "average_precision": average_precision_score(y_true, probability),
        "roc_auc": roc_auc_score(y_true, probability),
    }


def main() -> None:
    source = pd.read_csv(BASE_PATH)
    panel, numeric, categorical = prepare_panel(source)
    train = panel[panel["ano"] <= 2014].copy()
    tune = panel[panel["ano"] <= 2016].copy()
    test = panel[panel["ano"].between(2017, 2019)].copy()
    y_test = test["queda_10_pct"].to_numpy()

    tune_years = tune["ano"].to_numpy()
    cv = [
        (np.flatnonzero(tune_years < year), np.flatnonzero(tune_years == year))
        for year in (2015, 2016)
    ]
    cv = [(a, b) for a, b in cv if len(a) and len(b)]
    if len(cv) != 2:
        raise ValueError("Não foi possível criar os folds temporais completos de 2015 e 2016.")

    models = {
        "Random Forest": (
            make_pipeline(RandomForestClassifier(random_state=SEED, n_jobs=1, class_weight="balanced"), numeric, categorical),
            {
                "model__n_estimators": [200, 500],
                "model__max_depth": [6, 12],
                "model__min_samples_leaf": [2, 5],
            },
        ),
        "XGBoost": (
            make_pipeline(
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=SEED,
                    n_jobs=1,
                    tree_method="hist",
                ),
                numeric,
                categorical,
            ),
            {
                "model__n_estimators": [150, 300],
                "model__max_depth": [2, 4],
                "model__learning_rate": [0.03, 0.08],
                "model__subsample": [0.8],
                "model__colsample_bytree": [0.8],
            },
        ),
    }

    results = [{
        "modelo": "Sem queda (baseline)",
        **class_metrics(y_test, np.zeros(len(test), dtype=float)),
        "hiperparametros": "Regra fixa: todos os anos classificados sem queda",
    }]
    prediction_frames = []
    fitted_models = {}
    for name, (pipeline, grid) in models.items():
        print(f"Ajustando {name}...")
        search = GridSearchCV(
            pipeline,
            grid,
            scoring="average_precision",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(tune[numeric + categorical], tune["queda_10_pct"])
        probability = search.best_estimator_.predict_proba(test[numeric + categorical])[:, 1]
        results.append({
            "modelo": name,
            **class_metrics(y_test, probability),
            "hiperparametros": json.dumps(search.best_params_, ensure_ascii=False),
        })
        fitted_models[name] = search.best_estimator_
        prediction_frames.append(
            test[["codigo_municipio", "municipio", "ano", "variacao_alvo", "queda_10_pct"]]
            .assign(modelo=name, probabilidade_queda=probability, classe_prevista=(probability >= 0.5).astype(int))
        )

    metrics = pd.DataFrame(results)
    metrics.to_csv(TABLE_DIR / "metricas_alerta_queda.csv", index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(TABLE_DIR / "predicoes_alerta_queda.csv", index=False)

    prevalence = float(y_test.mean())
    cm_rows = []
    for name, fitted in fitted_models.items():
        subset = predictions[predictions["modelo"] == name]
        matrix = confusion_matrix(y_test, subset["classe_prevista"], labels=[0, 1])
        for actual, row in enumerate(matrix):
            for predicted, count in enumerate(row):
                cm_rows.append({"modelo": name, "real": actual, "previsto": predicted, "n": int(count)})
        fitted_importance = fitted.named_steps["model"].feature_importances_
        feature_names = fitted.named_steps["prep"].get_feature_names_out()
        imp = pd.DataFrame({"variable": feature_names, "importance": fitted_importance})
        imp["variable"] = imp["variable"].str.replace("num__", "", regex=False).str.replace("municipio__", "municipio_", regex=False)
        imp.sort_values("importance", ascending=False).head(15).to_csv(
            TABLE_DIR / f"importancia_{name.lower().replace(' ', '_')}.csv", index=False
        )
    pd.DataFrame(cm_rows).to_csv(TABLE_DIR / "matrices_confusion.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    for ax, name in zip(axes, ("Random Forest", "XGBoost")):
        matrix = confusion_matrix(
            y_test,
            predictions.loc[predictions["modelo"] == name, "classe_prevista"],
            labels=[0, 1],
        )
        ax.imshow(matrix, cmap="Blues")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([0, 1], ["Sem queda", "Queda"])
        ax.set_yticks([0, 1], ["Sem queda", "Queda"])
        ax.set_xlabel("Classe prevista", fontsize=7)
        ax.set_ylabel("Classe real", fontsize=7)
        ax.tick_params(labelsize=7)
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, str(value), ha="center", va="center", fontsize=9,
                    color="white" if value > matrix.max() * 0.55 else "black")
    fig.suptitle("Matrizes de confusão no teste temporal (2017-2019)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "matrices_confusion_test.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    best = metrics[metrics["modelo"] != "Sem queda (baseline)"].sort_values("average_precision", ascending=False).iloc[0]
    report_table = metrics.drop(columns="hiperparametros").rename(columns={
        "modelo": "Modelo",
        "precision": "Precisão",
        "recall": "Sensibilidade",
        "f1": "F1",
        "balanced_accuracy": "Acurácia balanceada",
        "average_precision": "Precisão média",
        "roc_auc": "ROC-AUC",
    }).to_markdown(index=False, floatfmt=".3f").replace(".", ",")
    n_observations = f"{len(panel):,}".replace(",", ".")
    n_test_observations = f"{len(test):,}".replace(",", ".")
    prevalence_text = f"{prevalence * 100:.1f}".replace(".", ",")
    report = f"""# Resultados: classificação de quedas na produção leiteira

## Pergunta e definição do evento

Classificou-se se a produção municipal no ano t diminuiu pelo menos 10% em relação a t-1. É uma tarefa de classificação de eventos, diferente da estimativa do volume anual em litros. O limiar é uma definição operacional do estudo, não um limite oficial de risco.

- Observações para modelagem: {n_observations}; municípios: {panel['codigo_municipio'].nunique()}.
- Treino/seleção: 2008-2016; folds de validação: 2015 e 2016; teste final: 2017-2019 ({n_test_observations} município-ano).
- Quedas >=10% no teste: {int(y_test.sum())} ({prevalence_text}%).
- Preditores: produção e crescimento disponíveis até t-1, ano, código municipal e clima de t-1. O clima do ano-alvo não é usado.
- Modelos: Random Forest e XGBoost; pesos de classe balanceados no RF; limiar de classificação fixo em 0,5.

## Resultados no teste

{report_table}

O modelo com maior average precision foi {best['modelo']} ({best['average_precision']:.3f}); a prevalência da classe positiva foi {prevalence:.3f}, referência para interpretar essa métrica. O limiar não foi modificado e o modelo não foi reajustado com os anos de teste.

## Saídas

- Tabelas: `results/decline_risk/tables/`
- Figura: `results/decline_risk/figures/matrices_confusion_test.png`
- Script reproduzível: `scripts/build_decline_risk_models.py`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"Eventos positivos no teste: {int(y_test.sum())}/{len(y_test)} ({prevalence:.1%})")
    print(f"Resultados: {REPORT_PATH}")


if __name__ == "__main__":
    main()
