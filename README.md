# CITREN Leite

Repositório do artigo **Predicción de la producción lechera municipal en Paraná mediante aprendizaje automático y datos agroclimáticos públicos**.

## Objetivo

Construir uma base pública integrada para previsão da produção anual de leite nos municípios do Paraná e comparar modelos de regressão baseados em:

- persistência temporal `lag-1`;
- Random Forest;
- XGBoost.

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
    figura_01_rmse_modelos.png
    figura_02_observado_predicho.png
    figura_03_series_municipios_lideres.png
    figura_04_importancia_variables.png

docs/
  articulo_CITREN_leite_materiales_resultados.docx
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

O período de teste foi separado temporalmente para evitar vazamento de informação futura. Os anos até 2016 foram usados para ajuste e validação interna; os anos de 2017 a 2019 foram reservados para avaliação final.

As métricas usadas foram:

- MAE;
- RMSE;
- R²;
- MAPE.

## Observação metodológica

O modelo usa uma estação meteorológica de Castro/PR como sinal agroclimático anual comum no painel municipal. Portanto, os resultados representam uma versão simplificada para congresso e podem ser ampliados em trabalhos futuros com dados climáticos espacializados, cobertura do solo, relevo e explicabilidade.
