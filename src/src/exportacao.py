"""Exportação de imagens do notebook e dos resultados finais."""

import base64
import json
from pathlib import Path

from . import config


def exportar_imagens_notebook(
    caminho_notebook,
    pasta_resultados=None,
    prefixo="grafico",
):
    """
    Extrai as imagens PNG armazenadas nas saídas das células do notebook.
    """
    caminho_notebook = Path(caminho_notebook)
    pasta = Path(pasta_resultados or config.PASTA_RESULTS)
    pasta.mkdir(parents=True, exist_ok=True)

    with caminho_notebook.open("r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    caminhos = []
    contador = 1

    for celula in dados["cells"]:
        for saida in celula.get("outputs", []):
            dados_saida = saida.get("data", {})

            if "image/png" not in dados_saida:
                continue

            imagem = dados_saida["image/png"]

            if isinstance(imagem, list):
                imagem = "".join(imagem)

            conteudo = base64.b64decode(imagem)
            destino = pasta / f"{prefixo}_{contador:03}.png"

            with destino.open("wb") as arquivo:
                arquivo.write(conteudo)

            caminhos.append(destino)
            contador += 1

    return caminhos


def exportar_metricas(
    comparacao_modelos=None,
    tabela_validacao_cruzada=None,
    tabela_validacao_grupos=None,
    tabela_modelos_otimizados=None,
    tabela_modelo_final=None,
    tabela_importancias=None,
    tabela_testes=None,
    pasta_metricas=None,
):
    """
    Exporta os mesmos grupos de resultados finais produzidos pelo notebook.
    Somente os objetos informados são gravados.
    """
    pasta = Path(pasta_metricas or config.PASTA_METRICAS)
    pasta.mkdir(parents=True, exist_ok=True)

    arquivos = {}

    if comparacao_modelos is not None:
        tabela = comparacao_modelos.copy()

        if tabela.index.name is not None or "Modelo" not in tabela.columns:
            tabela = tabela.rename_axis("Modelo").reset_index()

        destino = pasta / "comparacao_modelos.csv"
        tabela.to_csv(destino, index=False, encoding="utf-8-sig")
        arquivos["comparacao_modelos"] = destino

    if tabela_validacao_cruzada is not None:
        destino = pasta / "validacao_cruzada_modelos.csv"
        tabela_validacao_cruzada.to_csv(
            destino,
            index=False,
            encoding="utf-8-sig",
        )
        arquivos["validacao_cruzada_modelos"] = destino

    if tabela_validacao_grupos is not None:
        destino = pasta / "validacao_cruzada_grupos.csv"
        tabela_validacao_grupos.to_csv(
            destino,
            index=False,
            encoding="utf-8-sig",
        )
        arquivos["validacao_cruzada_grupos"] = destino

    if tabela_modelos_otimizados is not None:
        destino = pasta / "modelos_otimizados.csv"
        tabela_modelos_otimizados.to_csv(
            destino,
            index=False,
            encoding="utf-8-sig",
        )
        arquivos["modelos_otimizados"] = destino

    if tabela_modelo_final is not None:
        destino = pasta / "metricas_modelo_final.csv"
        tabela_modelo_final.to_csv(
            destino,
            index=False,
            encoding="utf-8-sig",
        )
        arquivos["metricas_modelo_final"] = destino

    if tabela_importancias is not None:
        destino = pasta / "importancia_variaveis.csv"
        tabela_importancias[
            ["Variável", "Importância (%)"]
        ].to_csv(
            destino,
            index=False,
            encoding="utf-8-sig",
        )
        arquivos["importancia_variaveis"] = destino

    if tabela_testes is not None:
        destino = pasta / "testes_estatisticos.csv"
        tabela_testes.to_csv(
            destino,
            index=False,
            encoding="utf-8-sig",
        )
        arquivos["testes_estatisticos"] = destino

    return arquivos
