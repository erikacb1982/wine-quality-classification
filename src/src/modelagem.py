"""Criação, treinamento e previsão dos modelos de classificação."""

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from .config import QUANTIDADE_VIZINHOS_KNN, RANDOM_STATE


def treinar_e_prever_modelo(
    modelo,
    caracteristicas_treino_modelo,
    classe_treino_modelo,
    caracteristicas_teste_modelo,
):
    """Treina o modelo e gera previsões para o conjunto de teste."""
    modelo.fit(
        caracteristicas_treino_modelo,
        classe_treino_modelo,
    )

    return modelo.predict(
        caracteristicas_teste_modelo
    )


def criar_regressao_logistica(
    random_state=RANDOM_STATE,
    max_iter=1000,
):
    """Cria a Regressão Logística com os parâmetros do notebook."""
    return LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
    )


def criar_random_forest(
    n_estimators=100,
    random_state=RANDOM_STATE,
):
    """Cria o Random Forest com os parâmetros iniciais do notebook."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )


def criar_knn(
    n_neighbors=QUANTIDADE_VIZINHOS_KNN,
):
    """Cria o KNN com 5 vizinhos por padrão."""
    return KNeighborsClassifier(
        n_neighbors=n_neighbors
    )


def criar_gradient_boosting(
    random_state=RANDOM_STATE,
):
    """Cria o Gradient Boosting com os parâmetros iniciais do notebook."""
    return GradientBoostingClassifier(
        random_state=random_state
    )


def criar_modelos_iniciais():
    """Retorna os quatro modelos avaliados no notebook."""
    return {
        "Regressão Logística": criar_regressao_logistica(),
        "Random Forest": criar_random_forest(),
        "KNN": criar_knn(),
        "Gradient Boosting": criar_gradient_boosting(),
    }


def treinar_modelos_iniciais(
    caracteristicas_treino,
    caracteristicas_teste,
    caracteristicas_treino_padronizadas,
    caracteristicas_teste_padronizadas,
    classe_treino,
):
    """
    Treina os quatro modelos respeitando a padronização usada
    no notebook: Regressão Logística e KNN usam dados padronizados.
    """
    modelos = criar_modelos_iniciais()
    previsoes = {}

    previsoes["Regressão Logística"] = treinar_e_prever_modelo(
        modelos["Regressão Logística"],
        caracteristicas_treino_padronizadas,
        classe_treino,
        caracteristicas_teste_padronizadas,
    )

    previsoes["Random Forest"] = treinar_e_prever_modelo(
        modelos["Random Forest"],
        caracteristicas_treino,
        classe_treino,
        caracteristicas_teste,
    )

    previsoes["KNN"] = treinar_e_prever_modelo(
        modelos["KNN"],
        caracteristicas_treino_padronizadas,
        classe_treino,
        caracteristicas_teste_padronizadas,
    )

    previsoes["Gradient Boosting"] = treinar_e_prever_modelo(
        modelos["Gradient Boosting"],
        caracteristicas_treino,
        classe_treino,
        caracteristicas_teste,
    )

    return modelos, previsoes
