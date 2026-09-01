"""Configurações gerais do projeto de classificação da qualidade de vinhos."""

from pathlib import Path

# ---------------------------------------------------------------------
# Caminhos do projeto
# ---------------------------------------------------------------------

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
PASTA_DADOS = RAIZ_PROJETO / "data"
CAMINHO_BASE_PADRAO = PASTA_DADOS / "WineQT.csv"

PASTA_RESULTS = RAIZ_PROJETO / "results"
PASTA_FIGURAS = PASTA_RESULTS / "figures"
PASTA_METRICAS = PASTA_RESULTS / "metrics"

# ---------------------------------------------------------------------
# Parâmetros usados no notebook
# ---------------------------------------------------------------------

LIMITE_ALTA_QUALIDADE = 7
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
QUANTIDADE_VIZINHOS_KNN = 5

# ---------------------------------------------------------------------
# Configuração visual
# ---------------------------------------------------------------------

CORES = {
    "principal": "#2F5D8C",
    "secundaria": "#7FA6C9",
    "contexto": "#C7CDD4",
    "destaque": "#4E7D5A",
    "texto": "#333333",
    "fundo": "#F8F9FA",
}

COR_HISTOGRAMA = "#4776A8"
COR_BOXPLOT = "#4776A8"

PALETA_BARRAS = [
    "#ABC7E0",
    "#759FC5",
    "#4776A8",
    "#56549A",
    "#673A8E",
    "#54206F",
]

CMAP_CONTINUO = "cividis"
CMAP_CONTINUO_ALTERNATIVO = "viridis"
CMAP_CORRELACAO = "coolwarm"

PALETA_CATEGORICA = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#000000",
]

COR_KDE = "#54206F"
COR_MEDIA = "#D55E00"
COR_MEDIANA = "#0072B2"

ALPHA_HISTOGRAMA = 0.85
ESPESSURA_KDE = 2.5

NOMES_VARIAVEIS = {
    "fixed acidity": "Acidez fixa",
    "volatile acidity": "Acidez volátil",
    "citric acid": "Ácido cítrico",
    "residual sugar": "Açúcar residual",
    "chlorides": "Cloretos",
    "free sulfur dioxide": "Dióxido de enxofre livre",
    "total sulfur dioxide": "Dióxido de enxofre total",
    "density": "Densidade",
    "pH": "pH",
    "sulphates": "Sulfatos",
    "alcohol": "Teor alcoólico",
    "quality": "Qualidade",
}

PRINCIPAIS_VARIAVEIS = [
    "alcohol",
    "volatile acidity",
    "sulphates",
]

# Mantém o comportamento original do notebook.
SALVAR_GRAFICOS = False
PASTA_GRAFICOS = "graficos"
DPI = 300
