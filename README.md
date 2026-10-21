# CITREN Leite

Repositório do artigo CITREN sobre previsão de produção leiteira municipal e do recorte independente preparado para o III WCAA.

## Objetivo

Construir uma base pública integrada para previsão da produção anual de leite nos municípios do Paraná e comparar modelos de regressão baseados em:

- persistência temporal `lag-1`;
- Random Forest;
- XGBoost.

O artigo do III WCAA usa uma pergunta diferente: classificar quedas anuais de pelo menos 10% com informação disponível até o ano anterior. As fontes públicas de base são parcialmente as mesmas, mas o alvo, a tarefa de aprendizado, os preditores sem vazamento temporal, as métricas e a conclusão são distintos. Os resultados do novo recorte são exploratórios e não indicam um alerta com desempenho operacional.

## Fontes de dados utilizadas

Foram usadas somente as bases efetivamente aplicadas ao artigo:

1. **IBGE/SIDRA - Pesquisa da Pecuária Municipal (PPM)**
   - Tabela SIDRA 74.
   - Produto 2682: leite.
   - Unidade: mil litros.
   - Link: https://sidra.ibge.gov.br/tabela/74
   - Uso: variável resposta de produção municipal anual de leite.

2. **INMET - Dados históricos da estação automática A819 Castro/PR**
   - Link: https://portal.inmet.gov.br/dadoshistoricos
   - Uso: variáveis agroclimáticas anuais agregadas a partir de dados horários.

## Estrutura

```text
scripts/
  build_article_models.py
  build_decline_risk_models.py
  create_conference_article.py

data/
  base_integrada_leite_agroclima.csv
  clima_castro_a819_anual.csv
  sidra_leite_pr_raw.json
  inmet_castro_a819_2006.csv ... inmet_castro_a819_2021.csv

results/
  tables/
    metricas_modelos.csv
    importancia_variables.csv
    predicciones_prueba.csv
  figures/
    figura_00_fluxo_metodologico.png
    figura_01_rmse_modelos.png
    figura_02_observado_predicho.png
    figura_03_series_municipios_lideres.png
    figura_04_importancia_variables.png
  decline_risk/
    tables/
    figures/
    resultados_alerta_queda.md

docs/
  Artigo_IIIWCAA_Alerta_Quedas_Leite.docx
```

## Como reproduzir

Instale as dependências principais:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib requests
```

Execute:

```bash
python scripts/build_article_models.py
```

O script gera a base integrada, as tabelas de métricas, as predições de teste, a importância das variáveis e as figuras usadas no artigo.

## Modelagem

O período de teste foi separado temporalmente para evitar vazamento de informação futura. Os anos de 2008 a 2016 foram usados para ajuste e validação interna; folds completos de 2015 e 2016 são validados com treinamento somente em anos anteriores. Os anos de 2017 a 2019 ficam reservados para avaliação final.

O script `scripts/create_conference_article.py` usa os resultados gerados e o DOCX-modelo III WCAA instalado em `C:\Users\gabri\Downloads\Artigo_IIIWCAA_NOME.docx` para compor o artigo, preservando o desenho do modelo e incorporando tabelas e figuras.

As métricas usadas foram:

- MAE;
- RMSE;
- R²;
- MAPE.

Para reexecutar a análise e regenerar o artigo, a partir da raiz do repositório:

```bash
python scripts/build_article_models.py
```

Gere o artigo no modelo do congresso:

```bash
python scripts/create_conference_article.py
```

## Observação metodológica

O modelo usa uma estação meteorológica de Castro/PR como sinal agroclimático anual comum no painel municipal. Portanto, os resultados representam uma versão simplificada para congresso e podem ser ampliados em trabalhos futuros com dados climáticos espacializados, cobertura do solo, relevo e explicabilidade.

## Recorte do III WCAA: classificação de quedas

Execute `python scripts/build_decline_risk_models.py` para reproduzir a tarefa de classificação. A classe positiva corresponde à redução de pelo menos 10% entre a produção do ano corrente e a do ano anterior. Os preditores climáticos também são defasados em um ano; clima do próprio ano-alvo não é usado. Random Forest e XGBoost são comparados a uma regra de referência que prevê sempre “sem queda”. Os anos 2017-2019 ficam reservados para teste, e as métricas incluem precisão, revocação, F1, acurácia balanceada, average precision e ROC-AUC.

Os resultados são gravados em `results/decline_risk/`. No teste observado, a prevalência de quedas foi 24,1%; os classificadores ficaram próximos do acaso e não devem ser tratados como alertas operacionais. O artigo de uma página é gerado com `python scripts/create_conference_article.py` usando o template III WCAA em `C:\Users\gabri\Downloads\Artigo_IIIWCAA_NOME.docx`.
