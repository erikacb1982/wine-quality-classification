"""Módulos do projeto Wine Quality Classification — Tech Challenge Fase 2."""

from .dados import carregar_dados, criar_variavel_alvo
from .preprocessamento import preparar_variaveis_modelagem

__all__ = [
    "carregar_dados",
    "criar_variavel_alvo",
    "preparar_variaveis_modelagem",
]
