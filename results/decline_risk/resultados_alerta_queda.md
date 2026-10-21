# Resultados: classificação de quedas na produção leiteira

## Pergunta e definição do evento

Classificou-se se a produção municipal no ano t diminuiu pelo menos 10% em relação a t-1. É uma tarefa de classificação de eventos, diferente da estimativa do volume anual em litros. O limiar é uma definição operacional do estudo, não um limite oficial de risco.

- Observações para modelagem: 4.788; municípios: 399.
- Treino/seleção: 2008-2016; folds de validação: 2015 e 2016; teste final: 2017-2019 (1.197 município-ano).
- Quedas >=10% no teste: 289 (24,1%).
- Preditores: produção e crescimento disponíveis até t-1, ano, código municipal e clima de t-1. O clima do ano-alvo não é usado.
- Modelos: Random Forest e XGBoost; pesos de classe balanceados no RF; limiar de classificação fixo em 0,5.

## Resultados no teste

| Modelo               |   Precisão |   Sensibilidade |    F1 |   Acurácia balanceada |   Precisão média |   ROC-AUC |
|:---------------------|-----------:|----------------:|------:|----------------------:|-----------------:|----------:|
| Sem queda (baseline) |      0,000 |           0,000 | 0,000 |                 0,500 |            0,241 |     0,500 |
| Random Forest        |      0,223 |           0,294 | 0,254 |                 0,484 |            0,236 |     0,488 |
| XGBoost              |      0,000 |           0,000 | 0,000 |                 0,498 |            0,242 |     0,500 |

O modelo com maior average precision foi XGBoost (0.242); a prevalência da classe positiva foi 0.241, referência para interpretar essa métrica. O limiar não foi modificado e o modelo não foi reajustado com os anos de teste.

## Saídas

- Tabelas: `results/decline_risk/tables/`
- Figura: `results/decline_risk/figures/matrices_confusion_test.png`
- Script reproduzível: `scripts/build_decline_risk_models.py`
