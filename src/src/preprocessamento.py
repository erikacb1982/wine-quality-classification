"""Preparação dos dados para modelagem."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE, TEST_SIZE


def preparar_variaveis_modelagem(df):
    """Separa características físico-químicas e variável alvo."""
    caracteristicas_vinho = df.drop(
        columns=[
            "Id",
            "quality",
            "quality_label",
        ]
    )
    classe_qualidade = df["quality_label"].copy()
    return caracteristicas_vinho, classe_qualidade


def resumo_separacao(caracteristicas_vinho, classe_qualidade):
    """Monta a tabela-resumo usada na preparação das variáveis."""
    quantidade_por_classe = classe_qualidade.value_counts()

    return pd.DataFrame({
        "Informação": [
            "Registros em X",
            "Variáveis explicativas",
            "Registros em y",
            "Baixa/Média Qualidade",
            "Alta Qualidade",
        ],
        "Resultado": [
            caracteristicas_vinho.shape[0],
            caracteristicas_vinho.shape[1],
            classe_qualidade.shape[0],
            quantidade_por_classe.get(0, 0),
            quantidade_por_classe.get(1, 0),
        ],
    })


def dividir_treino_teste(
    caracteristicas_vinho,
    classe_qualidade,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
):
    """
    Faz a divisão estratificada de 80% para treino e 20% para teste,
    seguindo os parâmetros usados no notebook.
    """
    return train_test_split(
        caracteristicas_vinho,
        classe_qualidade,
        test_size=test_size,
        random_state=random_state,
        stratify=classe_qualidade,
    )


def resumo_divisao(
    caracteristicas_vinho,
    caracteristicas_treino,
    caracteristicas_teste,
    classe_treino,
    classe_teste,
):
    """Retorna a tabela de conferência da divisão treino/teste."""
    return pd.DataFrame({
        "Conjunto": ["Treino", "Teste"],
        "Quantidade de Registros": [
            len(caracteristicas_treino),
            len(caracteristicas_teste),
        ],
        "Percentual da Base": [
            len(caracteristicas_treino) / len(caracteristicas_vinho) * 100,
            len(caracteristicas_teste) / len(caracteristicas_vinho) * 100,
        ],
        "Alta Qualidade (%)": [
            classe_treino.mean() * 100,
            classe_teste.mean() * 100,
        ],
    }).round(2)


def padronizar_variaveis(
    caracteristicas_treino,
    caracteristicas_teste,
):
    """
    Ajusta StandardScaler apenas no treino e aplica a transformação
    ao treino e ao teste.
    """
    padronizador = StandardScaler()

    treino_padronizado = padronizador.fit_transform(
        caracteristicas_treino
    )
    teste_padronizado = padronizador.transform(
        caracteristicas_teste
    )

    treino_padronizado = pd.DataFrame(
        treino_padronizado,
        columns=caracteristicas_treino.columns,
        index=caracteristicas_treino.index,
    )

    teste_padronizado = pd.DataFrame(
        teste_padronizado,
        columns=caracteristicas_teste.columns,
        index=caracteristicas_teste.index,
    )

    return treino_padronizado, teste_padronizado, padronizador


def comparar_padronizacao(
    caracteristicas_treino,
    caracteristicas_treino_padronizadas,
):
    """Compara média e desvio-padrão antes e depois da padronização."""
    return pd.DataFrame({
        "Média (Antes)": caracteristicas_treino.mean(),
        "Média (Depois)": caracteristicas_treino_padronizadas.mean(),
        "Desvio-Padrão (Antes)": caracteristicas_treino.std(ddof=0),
        "Desvio-Padrão (Depois)": (
            caracteristicas_treino_padronizadas.std(ddof=0)
        ),
    }).round(2)


def criar_grupos_registros(caracteristicas_vinho):
    """
    Cria o identificador de grupo usado na validação cruzada
    estratificada por grupos.
    """
    return (
        caracteristicas_vinho
        .astype(str)
        .agg("|".join, axis=1)
    )
