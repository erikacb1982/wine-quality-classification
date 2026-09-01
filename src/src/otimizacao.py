"""Otimização dos modelos candidatos com GridSearchCV."""

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import GridSearchCV, cross_validate

from .avaliacao import (
    METRICAS_VALIDACAO,
    criar_validacao_grupos,
)
from .config import RANDOM_STATE


PARAMETROS_RANDOM_FOREST = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

PARAMETROS_GRADIENT_BOOSTING = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.05, 0.10, 0.20],
    "max_depth": [1, 2, 3],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}


def otimizar_random_forest(
    caracteristicas_vinho,
    classe_qualidade,
    grupos,
    validacao_grupos=None,
    n_jobs=-1,
):
    """Executa o GridSearchCV do Random Forest com a grade do notebook."""
    validacao_grupos = (
        validacao_grupos
        if validacao_grupos is not None
        else criar_validacao_grupos()
    )

    modelo = RandomForestClassifier(
        random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=PARAMETROS_RANDOM_FOREST,
        scoring="f1",
        cv=validacao_grupos,
        n_jobs=n_jobs,
        return_train_score=True,
    )

    grid_search.fit(
        caracteristicas_vinho,
        classe_qualidade,
        groups=grupos,
    )

    return grid_search


def otimizar_gradient_boosting(
    caracteristicas_vinho,
    classe_qualidade,
    grupos,
    validacao_grupos=None,
    n_jobs=-1,
):
    """Executa o GridSearchCV do Gradient Boosting com a grade do notebook."""
    validacao_grupos = (
        validacao_grupos
        if validacao_grupos is not None
        else criar_validacao_grupos()
    )

    modelo = GradientBoostingClassifier(
        random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=PARAMETROS_GRADIENT_BOOSTING,
        scoring="f1",
        cv=validacao_grupos,
        n_jobs=n_jobs,
        return_train_score=True,
    )

    grid_search.fit(
        caracteristicas_vinho,
        classe_qualidade,
        groups=grupos,
    )

    return grid_search


def comparar_modelos_otimizados(
    grid_search_rf,
    grid_search_gb,
    caracteristicas_vinho,
    classe_qualidade,
    grupos,
    validacao_grupos=None,
):
    """Compara os dois melhores estimadores por validação cruzada em grupos."""
    validacao_grupos = (
        validacao_grupos
        if validacao_grupos is not None
        else criar_validacao_grupos()
    )

    modelos_otimizados = {
        "Random Forest Otimizado": grid_search_rf.best_estimator_,
        "Gradient Boosting Otimizado": grid_search_gb.best_estimator_,
    }

    resultados = []

    for nome_modelo, modelo in modelos_otimizados.items():
        resultado = cross_validate(
            modelo,
            caracteristicas_vinho,
            classe_qualidade,
            groups=grupos,
            cv=validacao_grupos,
            scoring=METRICAS_VALIDACAO,
        )

        resultados.append({
            "Modelo": nome_modelo,
            "Accuracy Média (%)": resultado["test_Accuracy"].mean() * 100,
            "Precision Média (%)": resultado["test_Precision"].mean() * 100,
            "Recall Médio (%)": resultado["test_Recall"].mean() * 100,
            "F1-Score Médio (%)": resultado["test_F1-Score"].mean() * 100,
            "F1-Score Desvio (%)": resultado["test_F1-Score"].std() * 100,
        })

    return (
        pd.DataFrame(resultados)
        .sort_values(by="F1-Score Médio (%)", ascending=False)
        .reset_index(drop=True)
    )


def resumo_grid_search(grid_search):
    """Retorna os melhores hiperparâmetros e o melhor F1 médio."""
    return {
        "Melhores hiperparâmetros": grid_search.best_params_,
        "Melhor F1-Score médio (%)": grid_search.best_score_ * 100,
    }
