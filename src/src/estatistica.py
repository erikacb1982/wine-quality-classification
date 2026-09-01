"""Cálculos estatísticos e análises descritivas do projeto."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

from . import config
from .visualizacao import salvar_grafico


def resumo_estatistico(df):
    """Reproduz o resumo estatístico das variáveis do notebook."""
    colunas_analise = df.columns.drop(
        ["Id", "quality_label"],
        errors="ignore",
    )

    return (
        df[colunas_analise]
        .describe()
        .T
        .round(4)
    )


def medidas_dispersao(df):
    """Calcula amplitude e coeficiente de variação."""
    colunas_analise = df.columns.drop(
        ["Id", "quality_label"],
        errors="ignore",
    )

    amplitude = (
        df[colunas_analise].max()
        - df[colunas_analise].min()
    )

    coeficiente_variacao = (
        df[colunas_analise].std()
        / df[colunas_analise].mean()
        * 100
    )

    return pd.DataFrame({
        "Amplitude": amplitude,
        "Coeficiente de Variação (%)": coeficiente_variacao,
    }).round(2)


def calcular_matriz_correlacao(df):
    """Calcula a correlação de Pearson entre as variáveis numéricas."""
    dados_correlacao = df.copy()

    dados_correlacao = dados_correlacao.drop(
        columns=["Id", "quality_label"],
        errors="ignore",
    )

    dados_numericos = dados_correlacao.select_dtypes(
        include="number"
    )

    return dados_numericos.corr(method="pearson")


def plotar_matriz_correlacao(
    matriz,
    mostrar=True,
    salvar=False,
    pasta=None,
):
    """Exibe a matriz de correlação em mapa de calor."""
    matriz_exibicao = matriz.rename(
        index=config.NOMES_VARIAVEIS,
        columns=config.NOMES_VARIAVEIS,
    )

    figura, ax = plt.subplots(figsize=(13, 9))

    sns.heatmap(
        matriz_exibicao,
        annot=True,
        fmt=".2f",
        cmap=config.CMAP_CORRELACAO,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Coeficiente de correlação"},
        ax=ax,
    )

    ax.set_title(
        "Matriz de Correlação das Variáveis",
        fontsize=14,
        fontweight="bold",
        color=config.CORES["texto"],
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()

    caminho = salvar_grafico(
        "matriz_correlacao",
        pasta=pasta,
        salvar=salvar,
        figura=figura,
    )

    if mostrar:
        plt.show()

    return figura, ax, caminho


def quantificar_outliers_iqr(df, colunas=None):
    """Quantifica outliers pelo critério de 1,5 x IQR."""
    if colunas is None:
        colunas = df.drop(
            columns=["Id", "quality", "quality_label"],
            errors="ignore",
        ).columns

    resultado_outliers = []

    for coluna in colunas:
        q1 = df[coluna].quantile(0.25)
        q3 = df[coluna].quantile(0.75)
        iqr = q3 - q1

        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        quantidade = (
            (df[coluna] < limite_inferior)
            | (df[coluna] > limite_superior)
        ).sum()

        percentual = quantidade / len(df) * 100

        resultado_outliers.append({
            "Variável": config.NOMES_VARIAVEIS.get(coluna, coluna),
            "Outliers": int(quantidade),
            "Percentual (%)": round(percentual, 2),
        })

    return (
        pd.DataFrame(resultado_outliers)
        .sort_values(by="Outliers", ascending=False)
        .reset_index(drop=True)
    )


def comparar_variaveis_por_classe(df, colunas=None):
    """Compara média e mediana das variáveis entre as duas classes."""
    if colunas is None:
        colunas = df.drop(
            columns=["Id", "quality", "quality_label"],
            errors="ignore",
        ).columns

    dados = df[list(colunas) + ["quality_label"]].copy()

    media_por_classe = (
        dados.groupby("quality_label")[list(colunas)]
        .mean()
        .T
    )

    mediana_por_classe = (
        dados.groupby("quality_label")[list(colunas)]
        .median()
        .T
    )

    tabela = pd.DataFrame({
        "Média — Baixa/Média": media_por_classe[0],
        "Média — Alta": media_por_classe[1],
        "Mediana — Baixa/Média": mediana_por_classe[0],
        "Mediana — Alta": mediana_por_classe[1],
    }).round(4)

    tabela.index = [
        config.NOMES_VARIAVEIS.get(coluna, coluna)
        for coluna in tabela.index
    ]

    return tabela


def teste_mann_whitney(
    df,
    colunas=None,
    alpha=0.05,
):
    """Executa o teste de Mann–Whitney U para as principais variáveis."""
    colunas = colunas or config.PRINCIPAIS_VARIAVEIS
    resultado_testes = []

    for coluna in colunas:
        valores_baixa_media = (
            df.loc[df["quality_label"] == 0, coluna]
            .dropna()
        )
        valores_alta = (
            df.loc[df["quality_label"] == 1, coluna]
            .dropna()
        )

        estatistica_u, valor_p = mannwhitneyu(
            valores_alta,
            valores_baixa_media,
            alternative="two-sided",
        )

        resultado_testes.append({
            "Variável": config.NOMES_VARIAVEIS.get(coluna, coluna),
            "Estatística U": estatistica_u,
            "p-valor": valor_p,
            "Diferença estatística (5%)": (
                "Sim" if valor_p < alpha else "Não"
            ),
        })

    return pd.DataFrame(resultado_testes)


def classificar_tamanho_efeito(valor):
    """Classifica a magnitude do tamanho de efeito usado no notebook."""
    magnitude = abs(valor)

    if magnitude < 0.10:
        return "Desprezível"
    if magnitude < 0.30:
        return "Pequeno"
    if magnitude < 0.50:
        return "Moderado"
    return "Grande"


def calcular_tamanho_efeito(df, colunas=None):
    """Calcula o tamanho de efeito derivado da estatística U."""
    colunas = colunas or config.PRINCIPAIS_VARIAVEIS
    resultado_efeito = []

    for coluna in colunas:
        valores_alta = (
            df.loc[df["quality_label"] == 1, coluna]
            .dropna()
        )
        valores_baixa_media = (
            df.loc[df["quality_label"] == 0, coluna]
            .dropna()
        )

        estatistica_u, _ = mannwhitneyu(
            valores_alta,
            valores_baixa_media,
            alternative="two-sided",
        )

        n_alta = len(valores_alta)
        n_baixa_media = len(valores_baixa_media)

        efeito = (
            (2 * estatistica_u)
            / (n_alta * n_baixa_media)
            - 1
        )

        resultado_efeito.append({
            "Variável": config.NOMES_VARIAVEIS.get(coluna, coluna),
            "Tamanho de efeito": efeito,
            "Magnitude": classificar_tamanho_efeito(efeito),
            "Direção": (
                "Maior em Alta"
                if efeito > 0
                else "Menor em Alta"
                if efeito < 0
                else "Sem direção"
            ),
        })

    return pd.DataFrame(resultado_efeito)


def calcular_faixas_observadas(df, colunas=None):
    """Calcula quartis, mediana, mínimo e máximo por classe."""
    colunas = colunas or config.PRINCIPAIS_VARIAVEIS
    resultado_faixas = []

    for coluna in colunas:
        for classe, nome_classe in [
            (0, "Baixa/Média"),
            (1, "Alta"),
        ]:
            valores = (
                df.loc[df["quality_label"] == classe, coluna]
                .dropna()
            )

            resultado_faixas.append({
                "Variável": config.NOMES_VARIAVEIS.get(coluna, coluna),
                "Classe": nome_classe,
                "Q1 (25%)": valores.quantile(0.25),
                "Mediana": valores.median(),
                "Q3 (75%)": valores.quantile(0.75),
                "Mínimo": valores.min(),
                "Máximo": valores.max(),
            })

    return pd.DataFrame(resultado_faixas)
