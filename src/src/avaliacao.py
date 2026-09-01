"""Métricas, validações e verificações dos modelos de classificação."""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    N_SPLITS,
    QUANTIDADE_VIZINHOS_KNN,
    RANDOM_STATE,
)


ORDEM_METRICAS = [
    "Acurácia",
    "Precision",
    "Recall",
    "F1-Score",
]

METRICAS_VALIDACAO = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1-Score": "f1",
}


def avaliar_classe_alta_qualidade(
    classe_real,
    previsoes,
    classe_positiva=1,
):
    """Calcula Precision, Recall e F1-Score para a classe positiva."""
    return {
        "Precision": precision_score(
            classe_real,
            previsoes,
            pos_label=classe_positiva,
            zero_division=0,
        ),
        "Recall": recall_score(
            classe_real,
            previsoes,
            pos_label=classe_positiva,
            zero_division=0,
        ),
        "F1-Score": f1_score(
            classe_real,
            previsoes,
            pos_label=classe_positiva,
            zero_division=0,
        ),
    }


def avaliar_modelo(
    classe_real,
    previsoes,
    classe_positiva=1,
):
    """Calcula as quatro métricas usadas no notebook."""
    resultados = avaliar_classe_alta_qualidade(
        classe_real,
        previsoes,
        classe_positiva=classe_positiva,
    )
    resultados["Acurácia"] = accuracy_score(
        classe_real,
        previsoes,
    )
    return resultados


def tabela_metricas(resultados):
    """Transforma um dicionário de métricas em tabela percentual."""
    return pd.DataFrame({
        "Métrica": ORDEM_METRICAS,
        "Resultado (%)": [
            resultados[metrica] * 100
            for metrica in ORDEM_METRICAS
        ],
    })


def avaliar_varios_modelos(classe_real, previsoes_por_modelo):
    """Avalia todas as previsões e retorna resultados e comparação."""
    resultados = {
        nome: avaliar_modelo(classe_real, previsoes)
        for nome, previsoes in previsoes_por_modelo.items()
    }

    comparacao = pd.DataFrame(resultados).T[
        ORDEM_METRICAS
    ] * 100

    return resultados, comparacao


def criar_modelos_validacao(
    quantidade_vizinhos_knn=QUANTIDADE_VIZINHOS_KNN,
):
    """Cria os modelos usados na validação cruzada do notebook."""
    return {
        "Regressão Logística": Pipeline([
            ("padronizacao", StandardScaler()),
            (
                "modelo",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                ),
            ),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
        "KNN": Pipeline([
            ("padronizacao", StandardScaler()),
            (
                "modelo",
                KNeighborsClassifier(
                    n_neighbors=quantidade_vizinhos_knn
                ),
            ),
        ]),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }


def validacao_cruzada_modelos(
    caracteristicas_treino,
    classe_treino,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
):
    """Executa StratifiedKFold com 5 divisões, como no notebook."""
    validacao = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    resultados_validacao = []

    for nome_modelo, modelo in criar_modelos_validacao().items():
        resultado = cross_validate(
            modelo,
            caracteristicas_treino,
            classe_treino,
            cv=validacao,
            scoring=METRICAS_VALIDACAO,
        )

        resultados_validacao.append({
            "Modelo": nome_modelo,
            "Accuracy Média (%)": resultado["test_Accuracy"].mean() * 100,
            "Precision Média (%)": resultado["test_Precision"].mean() * 100,
            "Recall Médio (%)": resultado["test_Recall"].mean() * 100,
            "F1-Score Médio (%)": resultado["test_F1-Score"].mean() * 100,
            "F1-Score Desvio (%)": resultado["test_F1-Score"].std() * 100,
        })

    return (
        pd.DataFrame(resultados_validacao)
        .sort_values(by="F1-Score Médio (%)", ascending=False)
        .reset_index(drop=True)
    )


def avaliar_baseline(
    caracteristicas_treino,
    classe_treino,
    caracteristicas_teste,
    classe_teste,
):
    """Treina e avalia DummyClassifier(strategy='most_frequent')."""
    modelo = DummyClassifier(
        strategy="most_frequent",
        random_state=RANDOM_STATE,
    )

    modelo.fit(
        caracteristicas_treino,
        classe_treino,
    )

    previsoes = modelo.predict(
        caracteristicas_teste
    )

    resultados = avaliar_modelo(
        classe_teste,
        previsoes,
    )

    return modelo, previsoes, resultados, tabela_metricas(resultados)


def verificar_registros_identicos(
    caracteristicas_treino,
    caracteristicas_teste,
    classe_teste=None,
):
    """Verifica registros do teste que também aparecem no conjunto de treino."""
    combinacoes_treino = set(
        caracteristicas_treino.itertuples(
            index=False,
            name=None,
        )
    )

    presente_no_treino = caracteristicas_teste.apply(
        lambda linha: tuple(linha) in combinacoes_treino,
        axis=1,
    )

    quantidade_teste = len(caracteristicas_teste)
    quantidade_repetidos = int(presente_no_treino.sum())
    percentual_repetidos = (
        quantidade_repetidos / quantidade_teste * 100
        if quantidade_teste
        else 0
    )

    resumo = pd.DataFrame({
        "Indicador": [
            "Registros no conjunto de teste",
            "Registros do teste também presentes no treino",
            "Registros exclusivos do conjunto de teste",
        ],
        "Quantidade": [
            quantidade_teste,
            quantidade_repetidos,
            quantidade_teste - quantidade_repetidos,
        ],
        "Percentual (%)": [
            100,
            percentual_repetidos,
            100 - percentual_repetidos,
        ],
    })

    por_classe = None

    if classe_teste is not None:
        analise = pd.DataFrame({
            "Classe": classe_teste,
            "Presente no Treino": presente_no_treino,
        })

        por_classe = (
            analise
            .groupby("Classe")["Presente no Treino"]
            .agg(["sum", "count"])
            .reset_index()
        )

        por_classe["Percentual (%)"] = (
            por_classe["sum"]
            / por_classe["count"]
        ) * 100

        por_classe = por_classe.rename(columns={
            "sum": "Registros presentes no treino",
            "count": "Total da classe",
        })

        por_classe["Classe"] = por_classe["Classe"].map({
            0: "Baixa/Média Qualidade",
            1: "Alta Qualidade",
        })

    return presente_no_treino, resumo, por_classe


def criar_validacao_grupos(
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
):
    """Cria o StratifiedGroupKFold usado no notebook."""
    return StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def validacao_cruzada_por_grupos(
    caracteristicas_vinho,
    classe_qualidade,
    grupos,
    validacao_grupos=None,
):
    """Executa a validação cruzada estratificada por grupos."""
    validacao_grupos = (
        validacao_grupos
        if validacao_grupos is not None
        else criar_validacao_grupos()
    )

    resultados_validacao = []

    for nome_modelo, modelo in criar_modelos_validacao().items():
        resultado = cross_validate(
            modelo,
            caracteristicas_vinho,
            classe_qualidade,
            groups=grupos,
            cv=validacao_grupos,
            scoring=METRICAS_VALIDACAO,
        )

        resultados_validacao.append({
            "Modelo": nome_modelo,
            "Accuracy Média (%)": resultado["test_Accuracy"].mean() * 100,
            "Precision Média (%)": resultado["test_Precision"].mean() * 100,
            "Recall Médio (%)": resultado["test_Recall"].mean() * 100,
            "F1-Score Médio (%)": resultado["test_F1-Score"].mean() * 100,
            "F1-Score Desvio (%)": resultado["test_F1-Score"].std() * 100,
        })

    return (
        pd.DataFrame(resultados_validacao)
        .sort_values(by="F1-Score Médio (%)", ascending=False)
        .reset_index(drop=True)
    )


def avaliar_modelo_final_por_grupos(
    modelo_final,
    caracteristicas_vinho,
    classe_qualidade,
    grupos,
    validacao_grupos=None,
    n_jobs=-1,
):
    """
    Gera previsões com cross_val_predict por grupos e calcula
    as métricas finais.
    """
    validacao_grupos = (
        validacao_grupos
        if validacao_grupos is not None
        else criar_validacao_grupos()
    )

    previsoes_finais = cross_val_predict(
        modelo_final,
        caracteristicas_vinho,
        classe_qualidade,
        groups=grupos,
        cv=validacao_grupos,
        method="predict",
        n_jobs=n_jobs,
    )

    resultados = avaliar_modelo(
        classe_qualidade,
        previsoes_finais,
    )

    return previsoes_finais, resultados, tabela_metricas(resultados)
