"""Carregamento, preparação inicial e verificações da base WineQT."""

from pathlib import Path

import pandas as pd

from .config import CAMINHO_BASE_PADRAO, LIMITE_ALTA_QUALIDADE


def carregar_dados(caminho_base=None):
    """
    Carrega a base WineQT.

    Se nenhum caminho for informado, utiliza data/WineQT.csv
    a partir da raiz do projeto.
    """
    caminho = Path(caminho_base) if caminho_base is not None else CAMINHO_BASE_PADRAO

    if not caminho.exists():
        raise FileNotFoundError(
            f"Base de dados não encontrada em: {caminho}"
        )

    return pd.read_csv(caminho)


def criar_variavel_alvo(df, limite=LIMITE_ALTA_QUALIDADE, copiar=True):
    """
    Cria quality_label seguindo a regra do notebook:
    quality >= 7 recebe classe 1; caso contrário, classe 0.
    """
    dados = df.copy() if copiar else df
    dados["quality_label"] = (dados["quality"] >= limite).astype(int)
    return dados


def distribuicao_variavel_original(df):
    """Retorna a distribuição da coluna quality."""
    return df["quality"].value_counts().sort_index()


def resumo_classes(df):
    """Retorna quantidade e percentual das classes de quality_label."""
    distribuicao = (
        df["quality_label"]
        .value_counts()
        .sort_index()
    )

    proporcao = (
        df["quality_label"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
    )

    tabela = pd.DataFrame({
        "Quantidade": distribuicao,
        "Percentual (%)": proporcao.round(2),
    })

    nomes = {
        0: "Baixa/Média Qualidade",
        1: "Alta Qualidade",
    }
    tabela.index = [nomes.get(indice, indice) for indice in tabela.index]
    return tabela


def dimensoes_base(df):
    """Retorna a quantidade de registros e de variáveis."""
    numero_registros, numero_variaveis = df.shape
    return {
        "Quantidade de registros": numero_registros,
        "Quantidade de variáveis": numero_variaveis,
    }


def verificar_valores_ausentes(df):
    """Retorna a quantidade de valores ausentes por coluna."""
    valores_ausentes = df.isnull().sum()
    return valores_ausentes[valores_ausentes > 0]


def total_valores_ausentes(df):
    """Retorna o total de valores ausentes na base."""
    return int(df.isna().sum().sum())


def verificar_duplicados(df):
    """Retorna o total de registros totalmente duplicados."""
    return int(df.duplicated().sum())


def verificar_duplicados_sem_id(df):
    """Verifica duplicidades desconsiderando a coluna identificadora Id."""
    return int(
        df.drop(columns=["Id"], errors="ignore")
        .duplicated()
        .sum()
    )


def obter_registros_repetidos(df):
    """Retorna todos os registros envolvidos em repetições, sem considerar Id."""
    colunas_analise = df.columns.drop("Id", errors="ignore")

    return (
        df[
            df.duplicated(
                subset=colunas_analise,
                keep=False,
            )
        ]
        .sort_values(by=list(colunas_analise))
    )


def verificar_consistencia_repetidos(df):
    """
    Identifica grupos com as mesmas características físico-químicas,
    mas com notas de quality diferentes.
    """
    caracteristicas = [
        coluna
        for coluna in df.columns
        if coluna not in ["Id", "quality", "quality_label"]
    ]

    notas_por_grupo = (
        df.groupby(caracteristicas)["quality"]
        .nunique()
    )

    return notas_por_grupo[notas_por_grupo > 1]


def verificar_valores_unicos(df):
    """Retorna a quantidade de valores únicos por coluna."""
    return df.nunique()
