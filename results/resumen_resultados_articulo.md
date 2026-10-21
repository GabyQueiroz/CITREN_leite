# Resultados para artículo

## Tema

**Predicción de la producción lechera municipal en Paraná mediante aprendizaje automático y datos agroclimáticos públicos**

## Base integrada

- Fuente agropecuaria: SIDRA/IBGE, Producción de la Pecuaria Municipal, leche, municipios de Paraná.
- Fuente climática: INMET, estación automática A819 - Castro/PR, históricos horarios agregados por año.
- Unidad analítica: municipio-año.
- Registros en la base integrada: 5,586.
- Periodo usado en modelado: 2008-2019, por disponibilidad de rezagos y clima.
- Prueba temporal: 2017-2019.

## Variables principales

- Producción de leche en mil litros.
- Rezagos: producción del año anterior, producción de dos años antes y media móvil de tres años.
- Variables climáticas anuales: precipitación acumulada, temperatura media, temperatura máxima observada, radiación media, humedad media, viento medio y días secos aproximados.
- Identificador municipal como efecto espacial tabular.

## Métricas en prueba

| modelo             |     MAE |    RMSE |       R2 |   MAPE_% | melhores_parametros                                                                                                                                                                               |
|:-------------------|--------:|--------:|---------:|---------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Persistencia lag-1 | 1139.22 | 2974.01 | 0.979952 |  20.0086 | -                                                                                                                                                                                                 |
| Random Forest      | 1520.69 | 3958.71 | 0.964478 |  23.4905 | {"regressor__model__max_depth": 8, "regressor__model__min_samples_leaf": 2, "regressor__model__n_estimators": 100}                                                                                |
| XGBoost            | 1662.51 | 6266.99 | 0.910975 |  23.3507 | {"regressor__model__colsample_bytree": 0.8, "regressor__model__learning_rate": 0.05, "regressor__model__max_depth": 4, "regressor__model__n_estimators": 300, "regressor__model__subsample": 0.8} |

## Mejor modelo de aprendizaje automático y comparación con baseline

El mejor algoritmo de aprendizaje automático por RMSE fue **Random Forest**, con R² = **0.964**, RMSE = **3958.71 mil litros** y MAPE = **23.49%**. Sin embargo, la persistencia lag-1 obtuvo RMSE = **2974.01 mil litros** y superó a RF y XGBoost en este periodo de prueba; por tanto, los algoritmos de ML no demostraron mejora predictiva frente al baseline.

## Archivos generados

- `data/base_integrada_leite_agroclima.csv`
- `results/tables/metricas_modelos.csv`
- `results/tables/predicciones_prueba.csv`
- `results/figures/figura_01_rmse_modelos.png`
- `results/figures/figura_02_observado_predicho.png`
- `results/figures/figura_03_series_municipios_lideres.png`
- `results/figures/figura_04_importancia_variables.png`
- `docs/Artigo_IIIWCAA_Predicao_Leite_Parana.docx`
