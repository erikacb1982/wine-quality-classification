"""Interpretação do modelo final e das principais variáveis."""

import pandas as pd

from . import config


def calcular_importancia_variaveis(
    modelo,
    caracteristicas_vinho,
    classe_qualidade,
):
    """Ajusta o modelo e monta a tabela de importância das variáveis."""
    modelo.fit(
        caracteristicas_vinho,
        classe_qualidade,
    )

    importancias = modelo.feature_importances_

    tabela = pd.DataFrame({
        "Variável Original": caracteristicas_vinho.columns,
        "Importância": importancias,
    })

    tabela["Variável"] = tabela["Variável Original"].map(
        lambda coluna: config.NOMES_VARIAVEIS.get(
            coluna,
            coluna,
        )
    )

    tabela["Importância (%)"] = (
        tabela["Importância"] * 100
    )

    return (
        tabela
        .sort_values(
            by="Importância (%)",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def comparar_principais_variaveis(
    df,
    colunas=None,
):
    """Compara média e mediana das principais variáveis entre as classes."""
    colunas = colunas or config.PRINCIPAIS_VARIAVEIS
    resultados = []

    for coluna in colunas:
        valores_baixa_media = df.loc[
            df["quality_label"] == 0,
            coluna,
        ]

        valores_alta = df.loc[
            df["quality_label"] == 1,
            coluna,
        ]

        resultados.append({
            "Variável": config.NOMES_VARIAVEIS.get(coluna, coluna),
            "Média — Baixa/Média": valores_baixa_media.mean(),
            "Média — Alta": valores_alta.mean(),
            "Mediana — Baixa/Média": valores_baixa_media.median(),
            "Mediana — Alta": valores_alta.median(),
        })

    return pd.DataFrame(resultados)


def adicionar_variacao_media(tabela_principais_variaveis):
    """Calcula a variação percentual da média da classe alta."""
    tabela = tabela_principais_variaveis.copy()

    tabela["Variação Média (%)"] = (
        (
            tabela["Média — Alta"]
            - tabela["Média — Baixa/Média"]
        )
        / tabela["Média — Baixa/Média"]
    ) * 100

    return tabela


def selecionar_principais_por_importancia(
    tabela_importancias,
    quantidade=3,
):
    """Retorna as variáveis mais importantes do modelo."""
    return tabela_importancias.head(quantidade).copy()
