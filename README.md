# Tech Challenge — Fase 2 | POSTECH Data Analytics

## 1. Identificação

| Campo | Valor |
|---|---|
| Turma | Data Analytics - 14DTAT |
| Grupo | Grupo 59 |
| Data de entrega | 31/08/2026 |

### Integrantes

| Nome completo | RM | E-mail |
|---|---|---|
| Erika Cristina Bernardo | RM374337 | |

---

## 2. Links da entrega

Estes três links são **obrigatórios** e devem ser idênticos aos do PDF de submissão.

## 2. Links da entrega

| Item | Link |
|---|---|
| Repositório | [GitHub - wine-quality-classification](https://github.com/erikacb1982/wine-quality-classification) |
| Vídeo executivo (≤ 5 min) | [YouTube](https://youtu.be/ahsa6PD0Ddg) |
| Apresentação | [Apresentação executiva](https://github.com/erikacb1982/wine-quality-classification/blob/main/data/apresentacao_executiva.pdf) |

---

## 3. O problema

A avaliação da qualidade de vinhos é tradicionalmente realizada por especialistas por meio de análises sensoriais, considerando características como aroma, sabor, acidez e equilíbrio. Embora importante, esse processo possui componentes subjetivos, pode demandar tempo e depende da experiência dos avaliadores.

Neste projeto, técnicas de **Machine Learning** são utilizadas para verificar se as características físico-químicas dos vinhos podem auxiliar na previsão de sua qualidade. A proposta é construir um modelo de classificação que possa apoiar a análise realizada por especialistas e contribuir para a identificação de padrões associados à qualidade do produto.

### Variável alvo

A variável alvo original é **`quality`**, que representa a nota de qualidade atribuída ao vinho.

Para transformar o problema em uma classificação binária, foi criada a variável **`quality_label`**, utilizando o seguinte critério:

- **Alta Qualidade (1):** `quality ≥ 7`
- **Baixa/Média Qualidade (0):** `quality < 7`

Após a transformação, a distribuição das classes ficou:

| Classe | Registros | Percentual |
|---|---:|---:|
| Baixa/Média Qualidade (0) | 984 | 86,09% |
| Alta Qualidade (1) | 159 | 13,91% |

A distribuição evidencia um **desbalanceamento entre as classes**, já que os vinhos classificados como Alta Qualidade representam apenas 13,91% da base.

Por esse motivo, a avaliação dos modelos não foi baseada apenas na acurácia. Também foram consideradas métricas como **Precision, Recall e F1-Score**, principalmente para avaliar a capacidade dos modelos de identificar corretamente os vinhos da classe Alta Qualidade.

### Dataset

| Campo | Valor |
|---|---|
| Fonte | [Kaggle — Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) |
| Linhas × colunas | 1.143 × 13 no arquivo original |
| Período / versão | Período temporal não informado; arquivo `WineQT.csv` utilizado no projeto |
| Licença de uso | CC0: Public Domain |

O conjunto utilizado possui **1.143 registros e 13 variáveis** no arquivo original. A variável `quality_label` é criada durante o processamento e, portanto, não pertence ao dataset bruto.

#### Descrição das variáveis

| Variável | Tipo | Descrição |
|---|---|---|
| `fixed acidity` | Numérica | Acidez fixa do vinho |
| `volatile acidity` | Numérica | Acidez volátil |
| `citric acid` | Numérica | Concentração de ácido cítrico |
| `residual sugar` | Numérica | Quantidade de açúcar residual |
| `chlorides` | Numérica | Concentração de cloretos |
| `free sulfur dioxide` | Numérica | Dióxido de enxofre livre |
| `total sulfur dioxide` | Numérica | Dióxido de enxofre total |
| `density` | Numérica | Densidade do vinho |
| `pH` | Numérica | Medida de acidez/alcalinidade |
| `sulphates` | Numérica | Concentração de sulfatos |
| `alcohol` | Numérica | Teor alcoólico |
| `quality` | Inteira | Nota de qualidade atribuída ao vinho |
| `Id` | Inteira | Identificador do registro |

A coluna **`Id`** é utilizada apenas como identificador e não participa da modelagem. A variável **`quality`** também é retirada das variáveis preditoras após a criação de `quality_label`, evitando que a informação utilizada para definir a própria classe alvo seja fornecida ao modelo.

## 4. Como reproduzir

```bash
git clone https://github.com/erikacb1982/wine-quality-classification.git
cd wine-quality-classification

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

O dataset utilizado no projeto está disponível em:

data/WineQT.csv

Depois de abrir o Jupyter Notebook, estão disponíveis duas versões da análise:

| # | Notebook | O que faz |
|---|---|---|
| 1 | `notebooks/Fase2_versao_final.ipynb` | Versão final utilizada na entrega acadêmica da Fase 2, contendo o fluxo completo da análise, desde a exploração dos dados até a avaliação dos modelos. |
| 2 | `notebooks/Fase2_versao_final_modular.ipynb` | Versão reorganizada posteriormente, utilizando módulos da pasta `src/` para melhorar a organização, reutilização e manutenção do código. |

O notebook `Fase2_versao_final.ipynb` corresponde à versão utilizada na entrega acadêmica.

O notebook `Fase2_versao_final_modular.ipynb` apresenta uma reorganização posterior do mesmo projeto, com parte do código distribuída em módulos reutilizáveis.

O experimento utiliza `random_state=42` nas etapas que envolvem aleatoriedade. Na separação inicial dos dados, foram utilizados 80% para treinamento e 20% para teste, com estratificação para preservar aproximadamente a proporção das classes.
---

## 5. Resultados
Comparação inicial dos modelos

A primeira comparação foi realizada utilizando a divisão estratificada de 80% para treino e 20% para teste.

| Modelo              | Acurácia | Precisão | Recall |     F1 |       AUC-ROC |
| ------------------- | -------: | -------: | -----: | -----: | ------------: |
| Regressão Logística |   86,46% |   52,38% | 34,38% | 41,51% | Não calculado |
| Random Forest       |   92,14% |   79,17% | 59,38% | 67,86% | Não calculado |
| KNN                 |   86,46% |   51,85% | 43,75% | 47,46% | Não calculado |
| Gradient Boosting   |   89,96% |   66,67% | 56,25% | 61,02% | Não calculado |


**Modelo escolhido:** No holdout inicial, o Random Forest apresentou os melhores resultados.
Entretanto, durante a análise foram identificados registros com características físico-químicas repetidas ou muito semelhantes. Por isso, também foi utilizada uma validação cruzada estratificada por grupos, mantendo combinações equivalentes dentro do mesmo grupo e reduzindo o risco de obter uma estimativa excessivamente otimista do desempenho. Nessa avaliação mais rigorosa, a diferença entre Random Forest e Gradient Boosting diminuiu consideravelmente.
Random Forest e Gradient Boosting seguiram para otimização de hiperparâmetros. Após o ajuste:

o Random Forest otimizado apresentou F1 médio de 45,63%;
o Gradient Boosting otimizado apresentou F1 médio de 49,97% e Recall médio de 44,05%.

O Gradient Boosting aumentou seu F1 médio de 44,97% para 49,97%, um ganho de 5,00 pontos percentuais após a otimização.
Na avaliação final out-of-fold por grupos, o modelo selecionado apresentou:
| Modelo final                |   Acurácia |   Precisão |     Recall |         F1 |       AUC-ROC |
| --------------------------- | ---------: | ---------: | ---------: | ---------: | ------------: |
| Gradient Boosting otimizado | **88,01%** | **59,32%** | **44,03%** | **50,54%** | Não calculado |

Modelo escolhido: Gradient Boosting otimizado — foi selecionado por apresentar o melhor equilíbrio entre F1-Score, Recall e estabilidade na validação por grupos após a otimização, com maior capacidade de reconhecer a classe minoritária quando comparado ao Random Forest otimizado.

**Métricas priorizadas:** F1-Score e Recall da classe Alta Qualidade. Como apenas 13,91% dos registros pertencem à classe positiva, uma acurácia elevada pode ocorrer mesmo quando o modelo possui dificuldade para identificar esses vinhos.

O Recall permite avaliar quantos dos vinhos realmente classificados como Alta Qualidade são encontrados pelo modelo. Já o F1-Score considera simultaneamente Recall e Precision, oferecendo uma medida de equilíbrio entre deixar de identificar vinhos de Alta Qualidade e classificar incorretamente vinhos de Baixa/Média como Alta.

O estudo de caso não estabelece um custo financeiro específico para falsos positivos e falsos negativos. Por isso, não foi assumido que um desses erros seja necessariamente mais caro que o outro, sendo adotado o F1-Score como principal critério de equilíbrio, com acompanhamento do Recall.

AUC-ROC não foi calculada na versão atual do projeto. O valor foi mantido como “Não calculado” para não apresentar uma métrica que não foi produzida pelo experimento.

## 6. Principais conclusões

1. As características físico-químicas apresentam informação útil para distinguir os níveis de qualidade, permitindo construir modelos capazes de apoiar a classificação dos vinhos. Entretanto, a identificação da classe Alta Qualidade permanece como o principal desafio: o modelo final apresentou Recall de 44,03%, indicando oportunidade de melhoria.
2. Teor alcoólico, acidez volátil e sulfatos foram as características de maior importância no Gradient Boosting, com 28,79%, 15,55% e 13,31%, respectivamente. Juntas, concentram 57,65% da importância atribuída pelo modelo.
3. Na base analisada, os vinhos classificados como Alta Qualidade tendem a apresentar maior teor alcoólico, menor acidez volátil e maior concentração de sulfatos. As diferenças observadas para essas três características também foram estatisticamente significativas. Esses resultados representam associações observadas na amostra e não demonstram causalidade.
4. A comparação entre a divisão tradicional treino/teste e a validação por grupos mostrou que a forma de validação influencia significativamente a estimativa de desempenho. A estratégia por grupos forneceu uma avaliação mais conservadora diante da existência de registros com características repetidas ou semelhantes.

### Limitações e próximos passos

A classe Alta Qualidade representa apenas 13,91% da base, o que dificulta seu aprendizado pelos modelos.
O conjunto contém somente 1.143 registros, limitando a variedade de exemplos disponíveis para treinamento e validação.
Existem registros com combinações físico-químicas repetidas ou muito semelhantes, exigindo cuidado para evitar resultados excessivamente otimistas.
A avaliação final é realizada sobre a própria base utilizada no desenvolvimento e na escolha dos hiperparâmetros, não constituindo um teste externo independente.
As relações identificadas entre características físico-químicas e qualidade são associações, não evidências de causa e efeito.
Como próximo passo, recomenda-se validar o modelo em novos dados independentes, testar estratégias específicas para o desbalanceamento das classes e avaliar o ajuste do limiar de classificação conforme o custo real de falsos positivos e falsos negativos no contexto de negócio.
A AUC-ROC poderá ser incorporada em uma evolução do projeto, utilizando as probabilidades previstas pelo modelo.

---

## 7. Estrutura do repositório

```text
wine-quality-classification/
│
├── data/
│   └── WineQT.csv
│
├── notebooks/
│   ├── Fase2_versao_final.ipynb
│   └── Fase2_versao_final_modular.ipynb
│
├── src/
│   └── módulos utilizados pela versão modularizada
│
├── results/
│   ├── figures/
│   └── metrics/
│
├── requirements.txt
└── README.md
```

---
## 8. Tecnologias

O projeto foi desenvolvido em Python e utiliza principalmente:

pandas — manipulação e análise dos dados;
NumPy — operações numéricas;
scikit-learn — pré-processamento, treinamento, validação, otimização e avaliação dos modelos;
SciPy — análises estatísticas;
Matplotlib — construção das visualizações;
Seaborn — visualizações estatísticas;
Jupyter Notebook — desenvolvimento e documentação da análise.

Os principais algoritmos avaliados foram:

Regressão Logística;
Random Forest;
K-Nearest Neighbors (KNN);
Gradient Boosting.

A modelagem também utiliza recursos como StandardScaler, validação cruzada estratificada, validação por grupos e GridSearchCV para otimização dos modelos.
