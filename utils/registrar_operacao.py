import duckdb
import os
from datetime import datetime
from colorama import Fore, Style, init
from typing import Optional
from utils.debug_logger import log_event
import numpy as np

init(autoreset=True)

# Caminho institucional para o banco DuckDB
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
db_path = os.path.join(base_path, "dados", "robodados.duckdb")

CAMPOS_TABELA = [
    "id", "timestamp", "ativo", "padrao", "regime", "contexto", "hora", "tipo",
    "volume", "preco_abertura", "preco_fechamento", "preco_saida", "lucro", "sinal", "resultado",
    "ticket", "retcode", "motivo_fechamento", "observacao", "motivo_saida", "score_reforco",
    "data_fechamento", "preco_entrada", "modelo_usado"
]

def sanity_check_dict(dados, colunas_check=None):
    """Checa valores nulos, NaN ou vazios em um dicionário. Retorna lista dos campos problemáticos."""
    if colunas_check is None:
        colunas_check = dados.keys()
    erros = []
    for k in colunas_check:
        v = dados.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == ""):
            erros.append(k)
    return erros

def registrar_operacao(
    ativo: str,
    ticket,
    padrao: str = "",
    regime: str = "",
    contexto: str = "",
    hora: str = "",
    motivo_fechamento: str = "",
    retcode: str = "",
    volume: float = 0.0,
    observacao: str = "",
    preco_abertura: float = None,
    preco_fechamento: float = None,
    preco_saida: float = None,
    lucro: float = None,
    resultado: float = None,
    sinal: int = None,
    timestamp=None,
    data_fechamento: Optional[str] = None,
    motivo_saida: str = "",
    score_reforco: float = None,
    preco_entrada: float = None,
    modelo_usado: str = ""
):
    """
    Registra uma operação no banco DuckDB, garantindo id incremental manual.
    Preenche TODOS os campos previstos na tabela 'operacoes'.
    """
    try:
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(timestamp, datetime):
            timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            timestamp = str(timestamp)

        if data_fechamento is not None:
            if isinstance(data_fechamento, datetime):
                data_fechamento = data_fechamento.strftime("%Y-%m-%d %H:%M:%S")
            else:
                data_fechamento = str(data_fechamento)

        dados_op = {
            "ativo": ativo,
            "ticket": ticket,
            "padrao": padrao,
            "regime": regime,
            "contexto": contexto,
            "hora": hora,
            "motivo_fechamento": motivo_fechamento,
            "retcode": retcode,
            "volume": volume,
            "observacao": observacao,
            "preco_abertura": preco_abertura,
            "preco_fechamento": preco_fechamento,
            "preco_saida": preco_saida,
            "lucro": lucro,
            "resultado": resultado,
            "sinal": sinal,
            "timestamp": timestamp,
            "data_fechamento": data_fechamento if data_fechamento else None,
            "motivo_saida": motivo_saida,
            "score_reforco": score_reforco,
            "preco_entrada": preco_entrada,
            "modelo_usado": modelo_usado,
            "tipo": "",  # Se quiser preencher depois, ajustar aqui
        }

        # SANITY CHECK — verifica se campos essenciais estão presentes e não nulos/vazios
        campos_essenciais = ["ativo", "ticket", "volume", "timestamp"]
        problemas = sanity_check_dict(dados_op, campos_essenciais)
        if problemas:
            msg = f"[SANITY CHECK] Campos essenciais faltando/invalidos ao registrar operação: {problemas}. Dados: {dados_op}"
            log_event(msg, level="warning")

        con = duckdb.connect(db_path)
        # Busca o próximo id incremental
        next_id = con.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM operacoes").fetchone()[0]
        params = [None]*len(CAMPOS_TABELA)
        params[0] = next_id

        # Preenche na ordem da tabela
        for idx, campo in enumerate(CAMPOS_TABELA[1:], 1):  # pula o id, já colocado
            params[idx] = dados_op.get(campo, None)

        sql = f"""
            INSERT INTO operacoes (
                {', '.join(CAMPOS_TABELA)}
            ) VALUES ({', '.join(['?']*len(CAMPOS_TABELA))})
        """
        con.execute(sql, params)
        con.close()
        msg = f"[OPERACAO] Operação registrada: id={next_id}, ativo={ativo}, ticket={ticket}, volume={volume}, sinal={sinal}"
        log_event(msg, level="info")
    except Exception as e:
        msg = f"[OPERACAO] Erro ao registrar operação: {e}"
        log_event(msg, level="error")

def atualizar_operacao(
    ticket,
    motivo_fechamento: Optional[str] = None,
    observacao: Optional[str] = None,
    preco_fechamento: Optional[float] = None,
    preco_saida: Optional[float] = None,
    lucro: Optional[float] = None,
    resultado: Optional[float] = None,
    sinal: Optional[int] = None,
    data_fechamento: Optional[str] = None,
    motivo_saida: Optional[str] = None,
    score_reforco: Optional[float] = None,
    preco_entrada: Optional[float] = None,
    modelo_usado: Optional[str] = None
):
    """
    Atualiza os principais campos de uma operação existente no banco.
    Agora também marca data_fechamento e campos expandidos.
    """
    try:
        con = duckdb.connect(db_path)
        updates = []
        params = []

        if motivo_fechamento is not None:
            updates.append("motivo_fechamento = ?")
            params.append(motivo_fechamento)
        if observacao is not None:
            updates.append("observacao = ?")
            params.append(observacao)
        if preco_fechamento is not None:
            updates.append("preco_fechamento = ?")
            params.append(preco_fechamento)
        if preco_saida is not None:
            updates.append("preco_saida = ?")
            params.append(preco_saida)
        if lucro is not None:
            updates.append("lucro = ?")
            params.append(lucro)
        if resultado is not None:
            updates.append("resultado = ?")
            params.append(resultado)
        if sinal is not None:
            updates.append("sinal = ?")
            params.append(sinal)
        if data_fechamento is None:
            data_fechamento = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updates.append("data_fechamento = ?")
        params.append(data_fechamento)
        if motivo_saida is not None:
            updates.append("motivo_saida = ?")
            params.append(motivo_saida)
        if score_reforco is not None:
            updates.append("score_reforco = ?")
            params.append(score_reforco)
        if preco_entrada is not None:
            updates.append("preco_entrada = ?")
            params.append(preco_entrada)
        if modelo_usado is not None:
            updates.append("modelo_usado = ?")
            params.append(modelo_usado)

        if not updates:
            msg = f"[OPERACAO] Nenhum dado para atualizar (ticket={ticket})"
            log_event(msg, level="warning")
            return

        sql = f"UPDATE operacoes SET {', '.join(updates)} WHERE ticket = ?"
        params.append(ticket)
        con.execute(sql, params)
        con.close()
        msg = f"[OPERACAO] Operação {ticket} atualizada no banco."
        log_event(msg, level="info")
    except Exception as e:
        msg = f"[OPERACAO] Erro ao atualizar operação: {e}"
        log_event(msg, level="error")
