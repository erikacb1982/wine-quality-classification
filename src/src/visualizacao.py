"""Funções e classe de visualização usadas na análise exploratória e avaliação."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from . import config


def adicionar_percentual(ax, total):
    """Adiciona o percentual acima de cada barra do gráfico."""
    for barra in ax.patches:
        altura = barra.get_height()
        percentual = altura / total * 100

        ax.annotate(
            f"{percentual:.1f}%",
            (barra.get_x() + barra.get_width() / 2, altura),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=config.CORES["texto"],
        )


def salvar_grafico(
    nome_arquivo,
    pasta=None,
    salvar=None,
    dpi=None,
    figura=None,
):
    """
    Salva o gráfico em PNG.

    Por padrão, respeita SALVAR_GRAFICOS definido em config.py.
    """
    if salvar is None:
        salvar = config.SALVAR_GRAFICOS

    if not salvar:
        return None

    pasta_destino = Path(pasta or config.PASTA_GRAFICOS)
    pasta_destino.mkdir(parents=True, exist_ok=True)

    dpi = dpi or config.DPI
    destino = pasta_destino / f"{nome_arquivo}.png"

    alvo = figura if figura is not None else plt
    alvo.savefig(
        destino,
        dpi=dpi,
        bbox_inches="tight",
    )
    return destino


def grafico_format(titulo, xlabel="", ylabel="", legenda=False):
    """Aplica a formatação visual padrão aos gráficos."""
    plt.title(
        titulo,
        fontsize=14,
        fontweight="bold",
        color=config.CORES["texto"],
    )
    plt.xlabel(
        xlabel,
        fontsize=11,
        color=config.CORES["texto"],
    )
    plt.ylabel(
        ylabel,
        fontsize=11,
        color=config.CORES["texto"],
    )
    plt.grid(
        axis="y",
        linestyle=":",
        alpha=0.20,
    )

    if legenda:
        plt.legend()

    plt.tight_layout()


class AnalisadorDistribuicao:
    """Analisa e representa graficamente a distribuição das variáveis numéricas."""

    def __init__(
        self,
        cor_histograma=config.COR_HISTOGRAMA,
        cor_kde=config.COR_KDE,
        cor_media=config.COR_MEDIA,
        cor_mediana=config.COR_MEDIANA,
        alpha_histograma=config.ALPHA_HISTOGRAMA,
        espessura_kde=config.ESPESSURA_KDE,
        bins=30,
    ):
        self.cor_histograma = cor_histograma
        self.cor_kde = cor_kde
        self.cor_media = cor_media
        self.cor_mediana = cor_mediana
        self.alpha_histograma = alpha_histograma
        self.espessura_kde = espessura_kde
        self.bins = bins

    def classificar_assimetria(self, assimetria):
        """Classifica a distribuição com base na assimetria."""
        intensidade = abs(assimetria)

        if intensidade < 0.50:
            return "Aproximadamente simétrica"
        if 0.50 <= assimetria < 1:
            return "Assimétrica à direita — intensidade moderada"
        if assimetria >= 1:
            return "Assimétrica à direita — intensidade elevada"
        if -1 < assimetria <= -0.50:
            return "Assimétrica à esquerda — intensidade moderada"
        return "Assimétrica à esquerda — intensidade elevada"

    def classificar_curtose(self, curtose):
        """Classifica a distribuição com base na curtose de Fisher."""
        if abs(curtose) < 0.50:
            return "Mesocúrtica"
        if curtose > 0:
            return "Leptocúrtica — caudas mais pesadas"
        return "Platicúrtica — caudas mais leves"

    def gerar_tabela_resumo(self, df, colunas, nomes_variaveis=None):
        """Gera média, mediana, assimetria, curtose e classificações."""
        resultados = []

        for coluna in colunas:
            if coluna not in df.columns:
                continue

            dados = df[coluna].dropna()
            if dados.empty:
                continue

            media = dados.mean()
            mediana = dados.median()
            assimetria = dados.skew()
            curtose = dados.kurt()

            nome_exibicao = (
                nomes_variaveis.get(coluna, coluna)
                if nomes_variaveis is not None
                else coluna
            )

            resultados.append({
                "Variável": nome_exibicao,
                "Média": media,
                "Mediana": mediana,
                "Assimetria": assimetria,
                "Classificação da Assimetria": self.classificar_assimetria(
                    assimetria
                ),
                "Curtose": curtose,
                "Classificação da Curtose": self.classificar_curtose(curtose),
            })

        tabela_resumo = pd.DataFrame(resultados)

        if tabela_resumo.empty:
            return tabela_resumo

        colunas_numericas = [
            "Média",
            "Mediana",
            "Assimetria",
            "Curtose",
        ]
        tabela_resumo[colunas_numericas] = (
            tabela_resumo[colunas_numericas].round(2)
        )
        return tabela_resumo

    def plotar_histograma(
        self,
        df,
        coluna,
        nome=None,
        mostrar=True,
        salvar=False,
        pasta=None,
    ):
        """Gera o histograma com KDE, média e mediana."""
        if coluna not in df.columns:
            raise KeyError(f"A variável '{coluna}' não foi encontrada.")

        dados = df[coluna].dropna()
        if dados.empty:
            raise ValueError(f"A variável '{coluna}' não possui valores válidos.")

        nome = nome or config.NOMES_VARIAVEIS.get(coluna, coluna)
        media = dados.mean()
        mediana = dados.median()

        figura, ax = plt.subplots(figsize=(9, 5))

        sns.histplot(
            data=dados,
            bins=self.bins,
            stat="count",
            kde=True,
            ax=ax,
            color=self.cor_histograma,
            alpha=self.alpha_histograma,
            edgecolor="white",
            linewidth=0.40,
        )

        if ax.lines:
            linha_kde = ax.lines[-1]
            linha_kde.set_color(self.cor_kde)
            linha_kde.set_linewidth(self.espessura_kde)
            linha_kde.set_label("Curva KDE")

        ax.axvline(
            x=media,
            color=self.cor_media,
            linestyle="--",
            linewidth=2,
            label=f"Média: {media:.2f}".replace(".", ","),
        )

        ax.axvline(
            x=mediana,
            color=self.cor_mediana,
            linestyle=":",
            linewidth=2.5,
            label=f"Mediana: {mediana:.2f}".replace(".", ","),
        )

        ax.set_axisbelow(True)

        grafico_format(
            titulo=f"Distribuição — {nome}",
            xlabel=nome,
            ylabel="Frequência",
            legenda=True,
        )

        caminho = salvar_grafico(
            f"histograma_{coluna}",
            pasta=pasta,
            salvar=salvar,
            figura=figura,
        )

        if mostrar:
            plt.show()

        return figura, ax, caminho


def plotar_distribuicao_alvo(
    df,
    mostrar=True,
    salvar=False,
    pasta=None,
):
    """Reproduz o gráfico de distribuição de quality_label do notebook."""
    figura, ax = plt.subplots(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="quality_label",
        hue="quality_label",
        palette=config.PALETA_CATEGORICA[:2],
        legend=False,
        ax=ax,
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        "Baixa/Média Qualidade",
        "Alta Qualidade",
    ])

    adicionar_percentual(ax, total=len(df))

    grafico_format(
        titulo="Distribuição das Classes de Qualidade",
        xlabel="Classe",
        ylabel="Quantidade",
    )

    caminho = salvar_grafico(
        "distribuicao_quality_label",
        pasta=pasta,
        salvar=salvar,
        figura=figura,
    )

    if mostrar:
        plt.show()

    return figura, ax, caminho


def plotar_boxplots_outliers(
    df,
    colunas,
    mostrar=True,
    salvar=False,
    pasta=None,
):
    """Gera os boxplots usados na análise visual de outliers."""
    figuras = {}

    for coluna in colunas:
        nome_variavel = config.NOMES_VARIAVEIS.get(coluna, coluna)
        figura, ax = plt.subplots(figsize=(9, 3.5))

        sns.boxplot(
            x=df[coluna],
            color=config.COR_BOXPLOT,
            ax=ax,
        )

        grafico_format(
            titulo=f"Boxplot — {nome_variavel}",
            xlabel=nome_variavel,
            ylabel="",
        )

        caminho = salvar_grafico(
            f"boxplot_{coluna}",
            pasta=pasta,
            salvar=salvar,
            figura=figura,
        )

        if mostrar:
            plt.show()

        figuras[coluna] = (figura, ax, caminho)

    return figuras


def plotar_comparacao_variaveis_classes(
    df,
    colunas,
    mostrar=True,
    salvar=False,
    pasta=None,
):
    """Gera os boxplots de cada variável por classe de qualidade."""
    dados = df[list(colunas) + ["quality_label"]].copy()
    dados["Classe"] = dados["quality_label"].map({
        0: "Baixa/Média Qualidade",
        1: "Alta Qualidade",
    })

    figuras = {}

    for coluna in colunas:
        nome_variavel = config.NOMES_VARIAVEIS.get(coluna, coluna)
        figura, ax = plt.subplots(figsize=(8, 4))

        sns.boxplot(
            data=dados,
            x="Classe",
            y=coluna,
            hue="Classe",
            palette=config.PALETA_CATEGORICA[:2],
            legend=False,
            ax=ax,
        )

        grafico_format(
            titulo=f"{nome_variavel} por Classe de Qualidade",
            xlabel="",
            ylabel=nome_variavel,
        )

        caminho = salvar_grafico(
            f"comparacao_classe_{coluna}",
            pasta=pasta,
            salvar=salvar,
            figura=figura,
        )

        if mostrar:
            plt.show()

        figuras[coluna] = (figura, ax, caminho)

    return figuras


def plotar_matriz_confusao(
    y_real,
    y_previsto,
    titulo,
    nome_arquivo=None,
    mostrar=True,
    salvar=False,
    pasta=None,
):
    """Gera uma matriz de confusão no padrão visual utilizado no notebook."""
    figura, eixo = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_real,
        y_previsto,
        display_labels=["Baixa/Média", "Alta"],
        cmap="Blues",
        values_format="d",
        colorbar=False,
        ax=eixo,
    )

    eixo.set_title(
        titulo,
        fontsize=14,
        fontweight="bold",
        color=config.CORES["texto"],
    )
    eixo.set_xlabel(
        "Classe Prevista",
        fontsize=11,
        color=config.CORES["texto"],
    )
    eixo.set_ylabel(
        "Classe Real",
        fontsize=11,
        color=config.CORES["texto"],
    )

    plt.tight_layout()

    caminho = None
    if nome_arquivo:
        caminho = salvar_grafico(
            nome_arquivo,
            pasta=pasta,
            salvar=salvar,
            figura=figura,
        )

    if mostrar:
        plt.show()

    return figura, eixo, caminho


def plotar_comparacao_modelos(
    tabela_comparacao,
    mostrar=True,
    salvar=False,
    pasta=None,
):
    """Gera o gráfico comparativo de Acurácia, Precision, Recall e F1-Score."""
    metricas = list(tabela_comparacao.columns)
    modelos = list(tabela_comparacao.index)

    x = np.arange(len(metricas))
    largura_barra = 0.18

    figura, eixo = plt.subplots(figsize=(11, 6))

    for indice, modelo in enumerate(modelos):
        deslocamento = (
            indice - (len(modelos) - 1) / 2
        ) * largura_barra

        barras = eixo.bar(
            x + deslocamento,
            tabela_comparacao.loc[modelo],
            width=largura_barra,
            label=modelo,
            color=config.PALETA_BARRAS[indice % len(config.PALETA_BARRAS)],
        )

        eixo.bar_label(
            barras,
            fmt="%.1f%%",
            padding=3,
            fontsize=9,
        )

    eixo.set_title(
        "Comparação de Desempenho dos Modelos",
        fontsize=14,
        fontweight="bold",
        color=config.CORES["texto"],
    )
    eixo.set_xlabel("Métricas", fontsize=11, color=config.CORES["texto"])
    eixo.set_ylabel("Resultado (%)", fontsize=11, color=config.CORES["texto"])
    eixo.set_xticks(x)
    eixo.set_xticklabels(metricas)
    eixo.set_ylim(0, 105)
    eixo.legend(title="Modelos")
    eixo.grid(axis="y", linestyle=":", alpha=0.20)
    eixo.set_axisbelow(True)

    plt.tight_layout()

    caminho = salvar_grafico(
        "comparacao_desempenho_modelos",
        pasta=pasta,
        salvar=salvar,
        figura=figura,
    )

    if mostrar:
        plt.show()

    return figura, eixo, caminho


def plotar_validacao_grupos(
    tabela_validacao_grupos,
    mostrar=True,
    salvar=True,
    pasta=None,
):
    """Gera o gráfico de comparação da validação cruzada por grupos."""
    pasta = Path(pasta or config.PASTA_FIGURAS)

    metricas = [
        "Accuracy Média (%)",
        "Precision Média (%)",
        "Recall Médio (%)",
        "F1-Score Médio (%)",
    ]
    nomes_metricas = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
    ]

    modelos = tabela_validacao_grupos["Modelo"].tolist()
    x = np.arange(len(metricas))
    largura_barra = 0.18

    figura, eixo = plt.subplots(figsize=(11, 6))

    for indice, modelo in enumerate(modelos):
        valores = (
            tabela_validacao_grupos
            .loc[tabela_validacao_grupos["Modelo"] == modelo, metricas]
            .iloc[0]
            .values
        )

        deslocamento = (
            indice - (len(modelos) - 1) / 2
        ) * largura_barra

        barras = eixo.bar(
            x + deslocamento,
            valores,
            width=largura_barra,
            label=modelo,
            color=config.PALETA_BARRAS[indice % len(config.PALETA_BARRAS)],
        )

        eixo.bar_label(
            barras,
            fmt="%.1f%%",
            padding=3,
            fontsize=8,
        )

    eixo.set_title(
        "Comparação dos Modelos — Validação Cruzada por Grupos",
        fontsize=14,
        fontweight="bold",
        color=config.CORES["texto"],
    )
    eixo.set_xlabel("Métricas", fontsize=11, color=config.CORES["texto"])
    eixo.set_ylabel(
        "Resultado Médio (%)",
        fontsize=11,
        color=config.CORES["texto"],
    )
    eixo.set_xticks(x)
    eixo.set_xticklabels(nomes_metricas)
    eixo.set_ylim(0, 100)
    eixo.legend(title="Modelos", frameon=False)

    plt.tight_layout()

    caminho = salvar_grafico(
        "comparacao_validacao_cruzada_grupos",
        pasta=pasta,
        salvar=salvar,
        figura=figura,
    )

    if mostrar:
        plt.show()

    return figura, eixo, caminho


def plotar_importancia_variaveis(
    tabela_importancias,
    mostrar=True,
    salvar=True,
    pasta=None,
):
    """Gera o gráfico de importância das variáveis do Gradient Boosting final."""
    pasta = Path(pasta or config.PASTA_FIGURAS)

    dados_grafico = (
        tabela_importancias[["Variável", "Importância (%)"]]
        .sort_values(by="Importância (%)", ascending=True)
    )

    figura, eixo = plt.subplots(figsize=(9, 6))

    barras = eixo.barh(
        dados_grafico["Variável"],
        dados_grafico["Importância (%)"],
        color=config.PALETA_BARRAS[0],
    )

    eixo.bar_label(
        barras,
        fmt="%.1f%%",
        padding=4,
        fontsize=9,
    )

    eixo.set_title(
        "Importância das Variáveis — Gradient Boosting Otimizado",
        fontsize=14,
        fontweight="bold",
        color=config.CORES["texto"],
    )
    eixo.set_xlabel(
        "Importância (%)",
        fontsize=11,
        color=config.CORES["texto"],
    )
    eixo.set_ylabel(
        "Variáveis",
        fontsize=11,
        color=config.CORES["texto"],
    )
    eixo.spines["top"].set_visible(False)
    eixo.spines["right"].set_visible(False)

    plt.tight_layout()

    caminho = salvar_grafico(
        "importancia_variaveis_gradient_boosting_final",
        pasta=pasta,
        salvar=salvar,
        figura=figura,
    )

    if mostrar:
        plt.show()

    return figura, eixo, caminho


def plotar_variacao_principais_variaveis(
    tabela_principais_variaveis,
    mostrar=True,
    salvar=False,
    pasta=None,
):
    """Gera o gráfico da variação média das principais variáveis entre classes."""
    tabela = tabela_principais_variaveis.copy()

    if "Variação Média (%)" not in tabela.columns:
        tabela["Variação Média (%)"] = (
            (
                tabela["Média — Alta"]
                - tabela["Média — Baixa/Média"]
            )
            / tabela["Média — Baixa/Média"]
        ) * 100

    variaveis = tabela["Variável"]
    valores = tabela["Variação Média (%)"]

    figura, ax = plt.subplots(figsize=(9, 5))

    barras = ax.bar(
        variaveis,
        valores,
        color=config.PALETA_BARRAS[0],
    )
    ax.axhline(y=0, linewidth=1)

    limite_superior = max(valores) + 6
    limite_inferior = min(valores) - 5
    ax.set_ylim(limite_inferior, limite_superior)

    for barra, valor in zip(barras, valores):
        rotulo = f"{valor:+.1f}%".replace(".", ",")
        x = barra.get_x() + barra.get_width() / 2

        if valor >= 0:
            ax.text(
                x,
                valor + 1.0,
                rotulo,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        else:
            ax.text(
                x,
                valor - 1.0,
                rotulo,
                ha="center",
                va="top",
                fontsize=11,
                fontweight="bold",
            )

    grafico_format(
        titulo=(
            "Variação Média das Principais Variáveis — "
            "Alta vs. Baixa/Média Qualidade"
        ),
        xlabel="",
        ylabel="Variação da média (%)",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    caminho = salvar_grafico(
        "variacao_principais_variaveis_classe",
        pasta=pasta,
        salvar=salvar,
        figura=figura,
    )

    if mostrar:
        plt.show()

    return figura, ax, caminho


def plotar_distribuicao_alvo_quantidade(
    df,
    mostrar=True,
    salvar=False,
    pasta=None,
):
    """Gera o gráfico de quantidade por classe da variável alvo."""
    figura, ax = plt.subplots(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="quality_label",
        color=config.COR_HISTOGRAMA,
        ax=ax,
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        "Baixa/Média\nQualidade",
        "Alta\nQualidade",
    ])

    for barra in ax.patches:
        altura = barra.get_height()
        ax.annotate(
            f"{int(altura)}",
            (
                barra.get_x() + barra.get_width() / 2,
                altura,
            ),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    grafico_format(
        titulo="Distribuição da Variável Alvo",
        xlabel="Classe",
        ylabel="Quantidade",
        legenda=False,
    )

    caminho = salvar_grafico(
        "distribuicao_variavel_alvo",
        pasta=pasta,
        salvar=salvar,
        figura=figura,
    )

    if mostrar:
        plt.show()

    return figura, ax, caminho


def plotar_boxplots_principais_ppt(
    df,
    variaveis=None,
    mostrar=True,
    salvar=True,
    pasta=None,
):
    """
    Reproduz os boxplots compactos das três variáveis principais
    usados para apoio à apresentação.
    """
    variaveis = variaveis or {
        "alcohol": "Teor alcoólico",
        "volatile acidity": "Acidez volátil",
        "sulphates": "Sulfatos",
    }

    pasta_destino = Path(pasta or config.PASTA_FIGURAS)
    pasta_destino.mkdir(parents=True, exist_ok=True)

    cor_baixa_media = "#0A2E61"
    cor_alta = "#00AEEF"
    figuras = {}

    for coluna, titulo in variaveis.items():
        baixa_media = df.loc[
            df["quality_label"] == 0,
            coluna,
        ].dropna()

        alta = df.loc[
            df["quality_label"] == 1,
            coluna,
        ].dropna()

        figura, ax = plt.subplots(figsize=(4.5, 4.2))

        box = ax.boxplot(
            [baixa_media, alta],
            tick_labels=["Baixa/Média", "Alta"],
            patch_artist=True,
            widths=0.55,
            medianprops={"color": "white", "linewidth": 2},
            boxprops={"linewidth": 1.2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "alpha": 0.35,
            },
        )

        box["boxes"][0].set_facecolor(cor_baixa_media)
        box["boxes"][1].set_facecolor(cor_alta)

        ax.set_title(
            titulo,
            fontsize=15,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle=":", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        caminho = None
        if salvar:
            nome_arquivo = (
                titulo.lower()
                .replace(" ", "_")
                .replace("ó", "o")
                .replace("á", "a")
                .replace("í", "i")
            )
            caminho = pasta_destino / f"{nome_arquivo}.png"
            figura.savefig(
                caminho,
                dpi=config.DPI,
                bbox_inches="tight",
                facecolor="white",
            )

        if mostrar:
            plt.show()

        figuras[coluna] = (figura, ax, caminho)

    return figuras
