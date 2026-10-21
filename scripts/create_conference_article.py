from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(r"C:\Users\gabri\Downloads\Artigo_IIIWCAA_NOME.docx")
OUT = ROOT / "docs" / "Artigo_IIIWCAA_Alerta_Quedas_Leite.docx"
METRICS_PATH = ROOT / "results" / "tables" / "metricas_modelos.csv"
PRED_PATH = ROOT / "results" / "tables" / "predicciones_prueba.csv"
IMPORTANCE_PATH = ROOT / "results" / "tables" / "importancia_variables.csv"
BASE_PATH = ROOT / "data" / "base_integrada_leite_agroclima.csv"
FIG = ROOT / "results" / "figures"
ALERT_DIR = ROOT / "results" / "decline_risk"


def fmt(x: float, digits: int = 2) -> str:
    return f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def run_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_params(value: str) -> str:
    params = json.loads(value)
    labels = {
        "n_estimators": "número de árvores",
        "max_depth": "profundidade máxima",
        "min_samples_leaf": "mínimo de amostras por folha",
        "learning_rate": "taxa de aprendizagem",
        "subsample": "proporção de linhas",
        "colsample_bytree": "proporção de atributos por árvore",
    }
    clean = {key.removeprefix("regressor__model__"): val for key, val in params.items()}
    return "; ".join(f"{labels.get(key, key)} = {val}" for key, val in clean.items())


def set_font(run, size=9, bold=None, italic=None):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def add_text(doc, text="", *, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=9.5, after=4, before=0):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(text)
    set_font(r, size=size, bold=bold, italic=italic)
    return p


def add_heading(doc, text):
    p = add_text(doc, text, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, size=10, before=5, after=3)
    p.paragraph_format.keep_with_next = True
    return p


def add_table(doc, headers, rows, widths=None, font_size=7.2):
    table = doc.add_table(rows=1, cols=len(headers))
    table.autofit = False
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:color"), "AAB8B8")
        borders.append(border)
    table._tbl.tblPr.append(borders)
    for i, value in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = str(value)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            for r in p.runs:
                set_font(r, font_size, bold=True)
                r.font.color.rgb = RGBColor(31, 78, 77)
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = str(value)
            cells[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cells[i].paragraphs:
                p.paragraph_format.space_after = Pt(0)
                for r in p.runs:
                    set_font(r, font_size)
    for index, row in enumerate(table.rows):
        row_props = row._tr.get_or_add_trPr()
        row_props.append(OxmlElement("w:cantSplit"))
        if index == 0:
            repeat_header = OxmlElement("w:tblHeader")
            repeat_header.set(qn("w:val"), "true")
            row_props.append(repeat_header)
    if widths:
        for row in table.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Inches(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_figure(doc, path: Path, caption: str):
    if not path.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(3.05))
    c = add_text(doc, caption, align=WD_ALIGN_PARAGRAPH.CENTER, size=8, after=4)
    c.paragraph_format.keep_together = True


def make_single_page_figure(metrics: pd.DataFrame, path: Path):
    ordered = metrics.sort_values("RMSE", ascending=True)
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    colors = ["#2a9d8f" if "Persistencia" in name else "#457b9d" if "Random Forest" in name else "#e76f51" for name in ordered["modelo"]]
    ax.barh(ordered["modelo"], ordered["RMSE"], color=colors, height=0.62)
    ax.set_xlabel("RMSE (mil litros)", fontsize=8)
    ax.set_title("Erro no teste temporal (2017-2019)", fontsize=9)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.6)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_one_page_body(doc, metrics, base, predictions):
    rf = metrics.loc[metrics["modelo"] == "Random Forest"].iloc[0]
    xgb = metrics.loc[metrics["modelo"] == "XGBoost"].iloc[0]
    baseline = metrics.loc[metrics["modelo"] == "Sem queda (baseline)"].iloc[0]
    prevalence = float(predictions.drop_duplicates(["codigo_municipio", "ano"])["queda_10_pct"].mean())

    add_text(doc, f"Resumo: Este estudo investiga um problema distinto da previsão de volumes: identificar municípios com queda anual de produção de leite igual ou superior a 10%. A partir da PPM/IBGE-SIDRA (tabela 74, produto 2682) e do painel agroclimático público já integrado, foram usados somente atributos disponíveis até t-1. Random Forest e XGBoost foram avaliados em teste temporal 2017-2019 (1.197 observações); a prevalência de quedas foi {fmt(prevalence*100,1)}%. Os resultados ficaram próximos da referência sem queda, indicando que esses dados não sustentaram um alerta confiável neste recorte.", size=8.5, after=3)
    add_text(doc, "Palavras-chave: produção leiteira; classificação; alerta de queda; Random Forest; XGBoost.", size=8.3, after=3)

    add_heading(doc, "1. Introdução")
    add_text(doc, "A produção municipal de leite pode variar entre anos, e reconhecer retrações expressivas é um problema diferente de estimar o volume produzido. Estudos de aprendizagem de máquina no setor leiteiro concentram-se sobretudo em predição produtiva e manejo, com desempenho dependente da escala e das variáveis observadas (Slob et al., 2021; Palma et al., 2025). Em vez de estimar um valor contínuo, este estudo define uma tarefa de classificação de quedas anuais, sem inferir causas.", size=8.4, after=3)

    add_heading(doc, "2. Materiais e Métodos")
    add_text(doc, f"A unidade foi município-ano, usando {base['codigo_municipio'].nunique()} municípios e a produção anual da PPM/IBGE, tabela SIDRA 74, produto leite 2682 (https://sidra.ibge.gov.br/tabela/74). O alvo é 1 quando (produção_t / produção_t-1) - 1 ≤ -0,10; o limiar de 10% é uma definição operacional deste estudo. O painel reutiliza fontes públicas da análise de regressão do grupo, mas muda a variável-resposta e a tarefa: classificação de quedas, não estimativa de volume. Os preditores foram produção em t-1 e t-2, crescimento até t-1, ano, código municipal e variáveis meteorológicas da estação INMET A819 de Castro agregadas ao ano anterior (https://portal.inmet.gov.br/dadoshistoricos). Não se usou clima do ano-alvo. A análise usa 2008-2019; dados ausentes nos lags necessários foram excluídos.", size=8.2, after=3)
    add_text(doc, "Compararam-se Random Forest (RF) e XGBoost (XGB) com baseline que classifica todos os casos como ‘sem queda’. Hiperparâmetros foram escolhidos por average precision em folds temporais completos de 2015 e 2016, sempre treinando em anos anteriores; o ajuste final usou dados até 2016 e teste intocado em 2017-2019. RF pesquisou 200/500 árvores, profundidade 6/12 e folha mínima 2/5, com pesos balanceados; selecionou 500, 12 e 5. XGB pesquisou 150/300 árvores, profundidade 2/4 e taxa 0,03/0,08, com amostragens 0,8; selecionou 150, 2 e 0,08. O limiar probabilístico foi fixo em 0,5. Métricas: precisão, sensibilidade (recall), F1, acurácia balanceada, average precision (AP) e ROC-AUC.", size=8.2, after=3)

    add_heading(doc, "3. Resultados e Discussão")
    metric_rows = []
    for model in ("Sem queda (baseline)", "Random Forest", "XGBoost"):
        row = metrics.loc[metrics["modelo"] == model].iloc[0]
        metric_rows.append([model, fmt(row["precision"],2), fmt(row["recall"],2), fmt(row["f1"],2), fmt(row["average_precision"],2), fmt(row["roc_auc"],2)])
    add_table(doc, ["Modelo", "Prec.", "Sens.", "F1", "AP", "AUC"], metric_rows, widths=[1.0, 0.4, 0.45, 0.35, 0.35, 0.4], font_size=6.2)
    add_text(doc, "Tabela 1. Métricas de classificação no teste temporal (2017-2019). Fonte: elaboração própria.", align=WD_ALIGN_PARAGRAPH.CENTER, size=6.8, after=1)
    add_figure(doc, ALERT_DIR / "figures" / "matrices_confusion_test.png", "Figura 1. Casos reais e sinalizados pelos classificadores com limiar 0,5. Fonte: elaboração própria.")
    add_text(doc, f"As quedas corresponderam a {fmt(prevalence*100,1)}% do teste. O RF identificou 85 dos 289 casos (sensibilidade={fmt(rf['recall'],2)}), mas gerou 296 falsos alertas e precisão de {fmt(rf['precision'],2)}; sua AP ({fmt(rf['average_precision'],3)}) ficou abaixo da referência ({fmt(baseline['average_precision'],3)}). O XGBoost obteve AP={fmt(xgb['average_precision'],3)}, praticamente igual à prevalência, e não sinalizou casos com o limiar fixo. As acurácias balanceadas foram {fmt(rf['balanced_accuracy'],3)} (RF) e {fmt(xgb['balanced_accuracy'],3)} (XGB); AUCs próximas de 0,5 indicam discriminação insuficiente. Os resultados não justificam uso operacional como alerta.", size=8.1, after=3)
    add_text(doc, "A regra ‘sem queda’ acertaria a maioria das observações por causa do desbalanceamento, mas teria sensibilidade nula; por isso, acurácia simples não seria evidência de alerta útil. O RF recuperou menos de um terço das retrações e mais de três quartos dos seus avisos foram falsos. A AP próxima da prevalência mostra que ordenar municípios por risco quase não melhorou em relação ao acaso. Em aplicações reais, falsos alertas também têm custo operacional e exigiriam validação com agentes do setor.", size=8.1, after=3)

    add_heading(doc, "4. Conclusión")
    add_text(doc, "Este recorte formula uma pergunta diferente da regressão do volume anual: avaliar detecção de retrações com atributos defasados e teste cronológico. Neste experimento, RF e XGBoost não superaram a referência simples; portanto, não se recomenda implantar alertas com estas variáveis. A PPM anual e o clima de uma única estação, replicado no estado, oferecem pouca informação espacial, e a queda de 10% é uma definição operacional sensível a revisões e flutuações. Estudos futuros devem incluir clima espacializado, tamanho e estrutura do rebanho e séries mais longas, validar o limiar com especialistas e avaliar sensibilidade, calibração e validação externa.", size=8.1, after=3)

    add_heading(doc, "Referências")
    refs = [
        "IBGE. Pesquisa da Pecuária Municipal, tabela SIDRA 74. https://sidra.ibge.gov.br/tabela/74. Acesso em: 20 out. 2026.",
        "INMET. Dados históricos de estações meteorológicas. https://portal.inmet.gov.br/dadoshistoricos. Acesso em: 20 out. 2026.",
        "PALMA, O. et al. AI and data analytics in the dairy farms: a scoping review. Animals, 15(9), 1291, 2025. doi:10.3390/ani15091291.",
        "SLOB, N.; CATAL, C.; KASSAHUN, A. Application of machine learning to improve dairy farm management. Preventive Veterinary Medicine, 187, 105237, 2021. doi:10.1016/j.prevetmed.2020.105237.",
    ]
    for ref in refs:
        add_text(doc, ref, align=WD_ALIGN_PARAGRAPH.LEFT, size=6.5, after=1)


def main():
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {TEMPLATE}")
    metrics_path = ALERT_DIR / "tables" / "metricas_alerta_queda.csv"
    predictions_path = ALERT_DIR / "tables" / "predicoes_alerta_queda.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        raise FileNotFoundError("Execute scripts/build_decline_risk_models.py antes de gerar o artigo.")

    metrics = pd.read_csv(metrics_path)
    pred = pd.read_csv(predictions_path)
    base = pd.read_csv(BASE_PATH)
    source_hash = run_hash(TEMPLATE)

    doc = Document(TEMPLATE)
    # Normalize fractional twips in this template so python-docx can calculate table widths.
    for section in doc.sections:
        margins = section._sectPr.find(qn("w:pgMar"))
        if margins is not None:
            for key, value in list(margins.attrib.items()):
                try:
                    margins.set(key, str(round(float(value))))
                except ValueError:
                    pass
    body = doc._element.body
    existing = list(body)
    # Retain the title/author area and its section break; remove only model prose below it.
    for child in existing[4:-1]:
        body.remove(child)

    doc.paragraphs[0].text = "CLASSIFICAÇÃO TEMPORAL DE QUEDAS NA PRODUÇÃO MUNICIPAL DE LEITE NO PARANÁ COM APRENDIZADO DE MÁQUINA"
    doc.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in doc.paragraphs[0].runs:
        set_font(run, 12, bold=True)
    authors = [
        "Andressa Aparecida Guimarães - Universidade Estadual de Ponta Grossa (UEPG)",
        "Gabrielly de Queiroz Pereira - Universidade Estadual de Ponta Grossa (UEPG)",
        "Maria Salete Marcon Gomes Vaz - Universidade Estadual de Ponta Grossa (UEPG)",
    ]
    for i, text in enumerate(authors, start=1):
        doc.paragraphs[i].text = text
        doc.paragraphs[i].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in doc.paragraphs[i].runs:
            set_font(run, 8.5, bold=True)

    build_one_page_body(doc, metrics, base, pred)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"Artigo de uma página criado: {OUT}")
    return

    add_text(doc, f"Resumo: Este trabalho avalia previsões anuais de um passo à frente da produção municipal de leite no Paraná por meio da integração de dados públicos agropecuários e meteorológicos. Foram relacionados os registros da Pesquisa da Pecuária Municipal (PPM/IBGE-SIDRA) e observações horárias da estação automática A819, em Castro/PR, agregadas por ano. O painel contém {n_rows:,} observações município-ano de {n_municipalities} municípios, e a modelagem foi realizada para 2008-2019 com atributos de produção defasada e clima anual. Compararam-se persistência (lag-1), Random Forest e XGBoost. A seleção de hiperparâmetros usou validação cronológica em anos completos de 2015 e 2016; 2017-2019 foi reservado como teste. O modelo de melhor desempenho entre os algoritmos de aprendizagem de máquina foi {best_name}, com MAE de {fmt(best_ml['MAE'])}, RMSE de {fmt(best_ml['RMSE'])} mil litros, R² de {fmt(best_ml['R2'],3)} e MAPE de {fmt(best_ml['MAPE_%'])}%. A persistência superou os dois algoritmos de ML no teste, indicando a força do histórico recente. O estudo oferece um recorte replicável para diagnóstico territorial, sem representar produção por fazenda nem relações causais.", size=9)
    add_text(doc, "Palavras-chave: produção leiteira; previsão municipal; Random Forest; XGBoost; dados agroclimáticos.", size=9, after=6)

    add_heading(doc, "1. Introdução")
    add_text(doc, "A previsão da produção de leite pode apoiar o planejamento agropecuário e a gestão de cadeias alimentares, e o aprendizado de máquina vem sendo aplicado a problemas de produção, qualidade e saúde animal (Slob et al., 2021; Palma et al., 2025). Estudos variam da produção diária em rebanhos à previsão nacional, com diferenças de escala, variáveis disponíveis e finalidade operacional (Samson et al., 2023; Cesarini et al., 2024).")
    add_text(doc, "Modelos baseados em árvores e métodos de boosting são úteis para representar relações não lineares; trabalhos com vacas individuais, contudo, indicam que o histórico de produção pode ser mais informativo que variáveis meteorológicas em alguns contextos (Breiman, 2001; Chen; Guestrin, 2016; Gasser et al., 2023). Assim, desempenho deve ser medido contra uma referência temporal simples e avaliado em períodos posteriores aos usados no ajuste.")
    add_text(doc, "Este artigo se diferencia ao focalizar o painel anual município-ano de todo o Paraná, em vez de dados de vacas ou propriedades, combinando PPM/IBGE e clima público da estação A819 de Castro. O clima é um sinal anual comum às observações municipais, não uma medição local para cada município. O objetivo é comparar persistência lag-1, Random Forest e XGBoost para previsão da produção municipal de leite, empregando validação cronológica e teste temporal independente.")

    add_heading(doc, "2. Materiais e Métodos")
    add_text(doc, "2.1 Área, unidade de análise e fontes de dados", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, size=9.5)
    add_text(doc, f"A unidade analítica é município-ano. A base agropecuária foi obtida na Pesquisa da Pecuária Municipal, tabela SIDRA 74, produto leite (código 2682), com produção em mil litros para os municípios paranaenses. Os dados meteorológicos vieram dos arquivos históricos do INMET para a estação automática A819, situada em Castro/PR; foram utilizadas precipitação, temperatura do ar, radiação global, umidade relativa e velocidade do vento quando disponíveis. O intervalo integrado efetivamente modelado cobre 2008-2019, após a criação de defasagens; a base integrada de entrada contém {n_rows:,} registros de {n_municipalities} municípios entre 2006 e 2019.")
    add_table(doc, ["Base", "Variáveis utilizadas", "Papel e endereço"], [
        ["IBGE/PPM-SIDRA 74", "Código e nome municipal; ano; produção de leite (mil litros)", "Alvo e histórico. https://sidra.ibge.gov.br/tabela/74"],
        ["INMET A819 - Castro", "Precipitação, temperatura, radiação, umidade, vento e dias secos aproximados, agregados por ano", "Preditores climáticos. https://portal.inmet.gov.br/dadoshistoricos"],
    ], widths=[0.82, 1.25, 1.0], font_size=6.5)
    add_text(doc, "Tabela 1. Fontes e variáveis efetivamente utilizadas. Fonte: elaboração própria.", align=WD_ALIGN_PARAGRAPH.CENTER, size=7.5, after=4)

    add_text(doc, "2.2 Integração e preparação", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, size=9.5)
    add_text(doc, "A série municipal do SIDRA foi convertida para tipos numéricos, padronizada por código IBGE e ano, e unida à tabela climática anual pela variável ano. As observações horárias do INMET foram interpretadas como dados brutos; valores sentinela (9999 e -9999) e ausências foram tratados como faltantes. Para cada ano calcularam-se precipitação acumulada, médias anuais das demais variáveis, máxima anual da temperatura horária e número aproximado de dias com precipitação diária inferior a 1 mm. A mesma série anual da estação A819 foi associada a todos os municípios em cada ano, portanto não há espacialização climática no painel.")
    add_text(doc, "Foram criados, dentro de cada município e em ordem temporal, produção defasada em um ano (lag-1), produção defasada em dois anos (lag-2), média móvel dos três anos anteriores e crescimento percentual entre as duas observações anteriores. A variável-resposta foi a produção do próprio ano. Registros sem os lags necessários foram excluídos; isso resultou em observações de 2008 a 2019. A imputação mediana e a padronização numérica, assim como a codificação one-hot dos identificadores municipais, foram ajustadas dentro do pipeline de cada treinamento, sem estimar parâmetros a partir do teste.")
    add_figure(doc, FIG / "figura_00_fluxo_metodologico.png", "Figura 1. Fluxo de integração, validação temporal e comparação dos modelos. Fonte: elaboração própria.")

    add_text(doc, "2.3 Modelos, hiperparâmetros e validação", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, size=9.5)
    add_text(doc, "A persistência lag-1 foi usada como baseline: a previsão para o ano t é a produção observada no ano t-1. O Random Forest (RF) constrói uma coleção de árvores de decisão sobre amostras e subconjuntos de atributos, agregando suas previsões (Breiman, 2001). O XGBoost (XGB) ajusta árvores sequencialmente para reduzir a função de perda, com regularização e amostragem de linhas/atributos (Chen; Guestrin, 2016). Para RF e XGB, o alvo foi transformado por log(1+y) durante o ajuste e reconvertido por exp(y)-1 na previsão, reduzindo a assimetria da escala de produção.")
    add_text(doc, "Os hiperparâmetros foram selecionados por GridSearchCV, minimizando RMSE médio nos anos de validação completos de 2015 e 2016. Em cada fold, o treinamento incluiu somente anos anteriores ao respectivo ano de validação. Após a escolha, o estimador foi reajustado com todos os dados de 2008-2016 e avaliado uma única vez em 2017-2019 (399 municípios × 3 anos = 1.197 observações). O conjunto de teste não participou da escolha de hiperparâmetros. A avaliação é de um passo à frente com atualização anual: para prever cada ano, o lag-1 corresponde à produção observada no ano imediatamente anterior; portanto, não se trata de uma projeção recursiva de três anos iniciada em 2016.")
    add_table(doc, ["Modelo", "Hiperparâmetros pesquisados"], [
        ["Random Forest", "n_estimators: 100, 300; max_depth: 8, sem limite; min_samples_leaf: 2; random_state=42"],
        ["XGBoost", "n_estimators: 100, 300; max_depth: 2, 4; learning_rate: 0,05; subsample=0,8; colsample_bytree=0,8; objective=squarederror; random_state=42"],
    ], widths=[0.9, 2.15], font_size=6.8)
    add_text(doc, "Tabela 2. Espaços de busca dos hiperparâmetros. Os valores escolhidos são apresentados no texto a seguir. Fonte: elaboração própria.", align=WD_ALIGN_PARAGRAPH.CENTER, size=7.5)
    for name in ("Random Forest", "XGBoost"):
        params = display_params(metrics.loc[name, "melhores_parametros"])
        add_text(doc, f"Parâmetros selecionados para {name}: {params}.", size=8.5, after=2)

    add_text(doc, "2.4 Métricas e implementação", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, size=9.5)
    add_text(doc, "A avaliação utiliza MAE (erro absoluto médio), RMSE (raiz do erro quadrático médio), R² (coeficiente de determinação) e MAPE (erro percentual absoluto médio). MAE e RMSE são expressos em mil litros; RMSE penaliza mais erros grandes. R² resume o ajuste conjunto e pode ser elevado pela variação estrutural de escala entre municípios, não devendo ser interpretado isoladamente. Para MAPE, o denominador foi limitado inferiormente a 1 mil litro para evitar divisão por zero. O processamento está em Python, com pandas, scikit-learn, XGBoost e matplotlib. Código e saídas estão em https://github.com/GabyQueiroz/CITREN_leite; os caminhos locais e o comando de reprodução são descritos no README do repositório.")

    add_heading(doc, "3. Resultados e Discussão")
    add_text(doc, f"A integração resultou em {n_rows:,} registros município-ano na base de entrada, cobrindo {n_municipalities} municípios e 2006-2019. Após a criação dos lags, a modelagem usou os anos de 2008 a 2019. O período de ajuste e seleção compreendeu {train_years}; a avaliação final compreendeu {test_years}, com {n_test:,} observações avaliadas por modelo.")
    metric_rows = []
    for _, row in metrics.iterrows():
        metric_rows.append([row["modelo"], fmt(row["MAE"]), fmt(row["RMSE"]), fmt(row["R2"], 3), f"{fmt(row['MAPE_%'])}%"])
    add_table(doc, ["Modelo", "MAE", "RMSE", "R²", "MAPE"], metric_rows, widths=[0.95, 0.52, 0.52, 0.48, 0.55], font_size=7)
    add_text(doc, "Tabela 3. Desempenho no teste temporal 2017-2019 (MAE e RMSE em mil litros). Fonte: resultados do experimento.", align=WD_ALIGN_PARAGRAPH.CENTER, size=7.5)
    add_figure(doc, FIG / "figura_01_rmse_modelos.png", "Figura 2. RMSE por modelo no período de teste 2017-2019; valores menores indicam menor erro quadrático. Fonte: elaboração própria.")
    add_figure(doc, FIG / "figura_02_observado_predicho.png", f"Figura 3. Produção observada e estimada pelo modelo de menor RMSE entre RF e XGB ({best_name}); a diagonal representa concordância perfeita. Fonte: elaboração própria.")

    gap = best_ml["RMSE"] - persistence["RMSE"]
    if gap < 0:
        discussion = f"Entre os modelos de aprendizagem de máquina, {best_name} obteve o menor RMSE ({fmt(best_ml['RMSE'])} mil litros). Sua comparação com a persistência é central: a diferença de RMSE foi {fmt(abs(gap))} mil litros, com vantagem para o modelo de ML. Isso sugere que os atributos municipais e agroclimáticos acrescentaram sinal preditivo ao lag-1 neste recorte, embora o desenho observacional não permita atribuir efeito causal ao clima."
    elif gap > 0:
        discussion = f"Embora {best_name} tenha sido o melhor dos dois algoritmos de ML, a persistência apresentou RMSE menor por {fmt(gap)} mil litros. Assim, o histórico recente de produção foi uma referência mais forte que os preditores adicionais neste teste. Esse achado é coerente com estudos que observam contribuição variável das condições meteorológicas quando a série produtiva passada já carrega informação temporal (Gasser et al., 2023; Cesarini et al., 2024); não se deve escolher o modelo apenas por R² ou por métricas altas em relação à média."
    else:
        discussion = f"O menor RMSE entre os algoritmos de ML foi obtido por {best_name}; seu resultado foi equivalente ao baseline de persistência nesta avaliação, portanto a evidência não demonstra ganho prático adicional dos preditores climáticos."
    add_text(doc, discussion)
    add_text(doc, f"O melhor modelo de ML apresentou MAE de {fmt(best_ml['MAE'])} mil litros, R² de {fmt(best_ml['R2'],3)} e MAPE de {fmt(best_ml['MAPE_%'])}%. MAE e RMSE expressam os erros na unidade do alvo, mas médias agregadas ocultam disparidades entre municípios de baixa e alta produção. O gráfico observado-predito deve ser lido com essa diferença de escala em mente; em dados municipais, um R² alto pode coexistir com erros absolutos relevantes para municípios menores. Como o lag é atualizado com o valor real do ano anterior, os resultados representam uma rotina anual de atualização e não uma previsão plurianual sem novos dados.")
    add_figure(doc, FIG / "figura_03_series_municipios_lideres.png", "Figura 4. Observações e previsões nos quatro municípios com maior produção média no conjunto de teste; a figura mostra somente 2017-2019, não uma reconstrução histórica. Fonte: elaboração própria.")
    if not importance.empty:
        imp_rows = [[r["variable"], fmt(r["importancia"], 4)] for _, r in importance.head(8).iterrows()]
        add_table(doc, ["Variável", "Importância"], imp_rows, widths=[1.8, 0.8], font_size=6.8)
        add_text(doc, f"Tabela 4. Oito maiores importâncias reportadas por {best_name}. A importância por impureza organiza a contribuição preditiva interna do modelo, mas não estabelece causalidade nem direção de efeito. Fonte: elaboração própria.", align=WD_ALIGN_PARAGRAPH.CENTER, size=7.3)
        add_figure(doc, FIG / "figura_04_importancia_variables.png", f"Figura 5. Importância relativa de atributos do modelo {best_name}; identificadores categóricos podem aparecer entre os preditores mais relevantes. Fonte: elaboração própria.")

    add_text(doc, "Em comparação com estudos de escala individual ou nacional, a contribuição aqui é o fluxo reprodutível de integração para todos os municípios paranaenses e a separação cronológica explícita de validação e teste (Samson et al., 2023; Cesarini et al., 2024). A agregação anual e municipal, porém, não permite inferir resposta de uma fazenda ou de uma vaca. Como a variável climática de Castro é replicada para todos os municípios em cada ano, a sua importância pode refletir associações temporais comuns, e não diferenças agroclimáticas espaciais. A série histórica do alvo também tende a dominar previsões de curto prazo, de modo que qualquer benefício do clima precisa ser avaliado contra o baseline e em validação temporal, como realizado neste trabalho.")

    add_heading(doc, "4. Conclusão")
    add_text(doc, f"O objetivo de comparar métodos de previsão anual municipal de leite no Paraná foi cumprido com dados públicos do IBGE/PPM e INMET e avaliação temporal independente. {best_name} foi o algoritmo de ML com menor RMSE, mas a comparação com a persistência mostrou que {('o algoritmo adicionou sinal preditivo neste recorte' if gap < 0 else 'os preditores adicionais não superaram o uso da produção do ano anterior neste recorte')}. Portanto, o resultado mais útil para aplicação é a necessidade de manter o baseline temporal e reportar erros em unidade produtiva, além de R².")
    add_text(doc, "As principais limitações são o número curto de anos no teste, a resolução anual da PPM, a dependência do histórico da variável-alvo e o uso de uma única estação meteorológica cujo sinal é comum ao estado no painel. Não há dados de produção por propriedade, manejo, rebanho, alimentação ou localização de produtores, e as métricas não estabelecem causalidade. Trabalhos futuros devem avaliar clima espacializado por município, séries mais recentes, validação rolling-origin em mais janelas, comparações com modelos estatísticos temporais e explicabilidade com SHAP ou importância por permutação, além de validação externa com registros locais de cooperativas ou laticínios.")
    add_heading(doc, "Agradecimentos")
    add_text(doc, "As autoras agradecem ao IBGE/SIDRA e ao INMET pela disponibilização pública das bases utilizadas.")

    add_heading(doc, "Referências")
    references = [
        "BREIMAN, L. Random forests. Machine Learning, v. 45, p. 5-32, 2001. DOI: https://doi.org/10.1023/A:1010933404324.",
        "CESARINI, L. et al. Comparison of deep learning models for milk production forecasting at national scale. Computers and Electronics in Agriculture, v. 221, 108933, 2024. DOI: https://doi.org/10.1016/j.compag.2024.108933.",
        "CHEN, T.; GUESTRIN, C. XGBoost: a scalable tree boosting system. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2016. p. 785-794. DOI: https://doi.org/10.1145/2939672.2939785.",
        "GASSER, L.; PEREZ CRUZ, F.; COCKBURN, M. Can meteorological data improve the short-term prediction of individual milk yield in dairy cows? Journal of Dairy Science, v. 106, n. 8, p. 5501-5516, 2023. DOI: https://doi.org/10.3168/jds.2022-22980.",
        "IBGE. Pesquisa da Pecuária Municipal: produção da pecuária municipal. Disponível em: https://www.ibge.gov.br/estatisticas/economicas/agricultura-e-pecuaria/9107-producao-da-pecuaria-municipal.html. Acesso em: 20 out. 2026.",
        "INMET. Dados históricos de estações meteorológicas. Disponível em: https://portal.inmet.gov.br/dadoshistoricos. Acesso em: 20 out. 2026.",
        "PALMA, O. et al. AI and data analytics in the dairy farms: a scoping review. Animals, v. 15, n. 9, 1291, 2025. DOI: https://doi.org/10.3390/ani15091291.",
        "SAMSON, T. K. et al. Smart modelling of dairy milk production with machine learning. IOP Conference Series: Earth and Environmental Science, v. 1219, 012008, 2023. DOI: https://doi.org/10.1088/1755-1315/1219/1/012008.",
        "SLOB, N.; CATAL, C.; KASSAHUN, A. Application of machine learning to improve dairy farm management: a systematic literature review. Preventive Veterinary Medicine, v. 187, 105237, 2021. DOI: https://doi.org/10.1016/j.prevetmed.2020.105237.",
    ]
    for ref in references:
        add_text(doc, ref, align=WD_ALIGN_PARAGRAPH.LEFT, size=7.6, after=3)
    add_text(doc, "Reprodutibilidade: script de modelagem em scripts/build_article_models.py; composição do DOCX em scripts/create_conference_article.py; base integrada em data/base_integrada_leite_agroclima.csv; tabelas em results/tables/; figuras em results/figures/. As saídas podem ser regeneradas conforme instruções no README.", italic=True, align=WD_ALIGN_PARAGRAPH.LEFT, size=7.2, after=1)

    # A continuous terminal section asks Word to balance the final page's two columns.
    final_section = doc.add_section(WD_SECTION_START.CONTINUOUS)
    final_cols = final_section._sectPr.find(qn("w:cols"))
    if final_cols is not None:
        final_cols.set(qn("w:num"), "1")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"Artigo criado: {OUT}")
    print(f"Template SHA-256: {source_hash}")


if __name__ == "__main__":
    main()
