# utils/registrar_operacao.py
# Registrador institucional com:
#  - Garantia de schema no DuckDB (CREATE/ADD COLUMN IF NOT EXISTS)
#  - timestamp como TIMESTAMPTZ e data_fechamento TIMESTAMP
#  - Fallback CSV se o banco falhar (sem perder o evento)
#  - Sanity check de campos essenciais

import duckdb
import os
from datetime import datetime, timezone
from colorama import Fore, Style, init
from typing import Optional, List, Dict, Any
import numpy as np

from utils.debug_logger import log_event

init(autoreset=True)

# Caminho institucional para o banco DuckDB
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
db_path = os.path.join(base_path, "dados", "robodados.duckdb")
fallback_csv = os.path.join(base_path, "dados", "operacoes_fallback.csv")

CAMPOS_TABELA: List[str] = [
    "id", "timestamp", "ativo", "padrao", "regime", "contexto", "hora", "tipo",
    "volume", "preco_abertura", "preco_fechamento", "preco_saida", "lucro", "sinal", "resultado",
    "ticket", "retcode", "motivo_fechamento", "observacao", "motivo_saida", "score_reforco",
    "data_fechamento", "preco_entrada", "modelo_usado"
]

# Tipagem alvo no DuckDB (colunas não listadas aqui viram VARCHAR por padrão)
TIPOS_TABELA: Dict[str, str] = {
    "id": "BIGINT",
    "timestamp": "TIMESTAMPTZ",
    "ativo": "VARCHAR",
    "padrao": "VARCHAR",
    "regime": "VARCHAR",
    "contexto": "VARCHAR",
    "hora": "VARCHAR",
    "tipo": "VARCHAR",
    "volume": "DOUBLE",
    "preco_abertura": "DOUBLE",
    "preco_fechamento": "DOUBLE",
    "preco_saida": "DOUBLE",
    "lucro": "DOUBLE",
    "sinal": "INTEGER",
    "resultado": "DOUBLE",
    "ticket": "VARCHAR",
    "retcode": "VARCHAR",
    "motivo_fechamento": "VARCHAR",
    "observacao": "VARCHAR",
    "motivo_saida": "VARCHAR",
    "score_reforco": "DOUBLE",
    "data_fechamento": "TIMESTAMP",
    "preco_entrada": "DOUBLE",
    "modelo_usado": "VARCHAR",
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _ensure_dados_dir() -> None:
    d = os.path.join(base_path, "dados")
    os.makedirs(d, exist_ok=True)

def _to_iso_ts(val) -> str:
    """
    Converte para string ISO (UTC) amigável ao TIMESTAMPTZ.
    Aceita datetime naive/aware, str, epoch.
    """
    if isinstance(val, datetime):
        dt = val if val.tzinfo else val.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    # epoch numérico em segundos ou ms
    try:
        if isinstance(val, (int, float)):
            x = float(val)
            if x > 1e12:
                x = x / 1000.0
            return datetime.fromtimestamp(x, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    except Exception:
        pass
    # string — retorna como veio (DuckDB tentará converter)
    return str(val)

def _ensure_schema(con: "duckdb.DuckDBPyConnection") -> None:
    """
    Garante que a tabela 'operacoes' exista com as colunas necessárias.
    Se não existir, cria. Se faltar coluna, adiciona.
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS operacoes (
            id BIGINT,
            timestamp TIMESTAMPTZ,
            ativo VARCHAR,
            padrao VARCHAR,
            regime VARCHAR,
            contexto VARCHAR,
            hora VARCHAR,
            tipo VARCHAR,
            volume DOUBLE,
            preco_abertura DOUBLE,
            preco_fechamento DOUBLE,
            preco_saida DOUBLE,
            lucro DOUBLE,
            sinal INTEGER,
            resultado DOUBLE,
            ticket VARCHAR,
            retcode VARCHAR,
            motivo_fechamento VARCHAR,
            observacao VARCHAR,
            motivo_saida VARCHAR,
            score_reforco DOUBLE,
            data_fechamento TIMESTAMP,
            preco_entrada DOUBLE,
            modelo_usado VARCHAR
        )
    """)
    # Adiciona colunas ausentes (tipos alvo se conhecidos)
    cols = con.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'operacoes'
    """).fetchall()
    existentes = {c[0] for c in cols}
    for c in CAMPOS_TABELA:
        if c not in existentes:
            tipo = TIPOS_TABELA.get(c, "VARCHAR")
            con.execute(f'ALTER TABLE operacoes ADD COLUMN "{c}" {tipo}')

def _sanity_list(dados: Dict[str, Any], campos: List[str]) -> List[str]:
    erros = []
    for k in campos:
        v = dados.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == ""):
            erros.append(k)
    return erros

def _write_csv_fallback(record: Dict[str, Any], acao: str = "insert") -> None:
    """
    Persistência de último recurso: registra a operação em CSV.
    """
    try:
        _ensure_dados_dir()
        header_needed = not os.path.exists(fallback_csv)
        import csv
        with open(fallback_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["acao"] + CAMPOS_TABELA)
            if header_needed:
                w.writeheader()
            row = {"acao": acao}
            # garante todas as chaves
            for k in CAMPOS_TABELA:
                row[k] = record.get(k, None)
            w.writerow(row)
        log_event(f"[OPERACAO/CSV] Fallback {acao} registrado em {fallback_csv}", level="warning")
    except Exception as e:
        log_event(f"[OPERACAO/CSV] Erro no fallback {acao}: {e}", level="error")

# ---------------------------------------------------------------------
# API
# ---------------------------------------------------------------------
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
    # Monta dict de dados (mantendo compatibilidade com sua assinatura)
    if timestamp is None:
        timestamp_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    else:
        timestamp_iso = _to_iso_ts(timestamp)

    if data_fechamento is not None:
        try:
            if isinstance(data_fechamento, datetime):
                data_fech_str = data_fechamento.strftime("%Y-%m-%d %H:%M:%S")
            else:
                data_fech_str = str(data_fechamento)
        except Exception:
            data_fech_str = None
    else:
        data_fech_str = None

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
        "timestamp": timestamp_iso,
        "data_fechamento": data_fech_str,
        "motivo_saida": motivo_saida,
        "score_reforco": score_reforco,
        "preco_entrada": preco_entrada,
        "modelo_usado": modelo_usado,
        "tipo": "",  # mantido como no seu arquivo (preenchido externamente)
    }

    # SANITY CHECK — verifica campos essenciais
    essenciais = ["ativo", "ticket", "volume", "timestamp"]
    problemas = _sanity_list(dados_op, essenciais)
    if problemas:
        log_event(f"[SANITY CHECK] Campos essenciais faltando/invalidos ao registrar: {problemas}. Dados={dados_op}", level="warning")

    # Tenta gravar no DuckDB; se falhar, escreve fallback CSV
    try:
        _ensure_dados_dir()
        con = duckdb.connect(db_path)
        _ensure_schema(con)

        # Busca o próximo id incremental
        next_id = con.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM operacoes").fetchone()[0]

        params = [None] * len(CAMPOS_TABELA)
        params[0] = next_id  # id

        for idx, campo in enumerate(CAMPOS_TABELA[1:], 1):
            params[idx] = dados_op.get(campo, None)

        placeholders = ", ".join(["?"] * len(CAMPOS_TABELA))
        sql = f"INSERT INTO operacoes ({', '.join(CAMPOS_TABELA)}) VALUES ({placeholders})"
        con.execute(sql, params)
        con.close()

        log_event(f"[OPERACAO] Operação registrada: id={next_id}, ativo={ativo}, ticket={ticket}, volume={volume}, sinal={sinal}", level="info")

    except Exception as e:
        log_event(f"[OPERACAO] Erro ao registrar operação no DuckDB: {e}. Gravando fallback CSV.", level="error")
        # Prepara linha completa para fallback
        record = {k: None for k in CAMPOS_TABELA}
        record.update(dados_op)
        # tenta carregar id anterior do CSV, senão deixa em branco
        record["id"] = None
        _write_csv_fallback(record, acao="insert")

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
    modelo_usado: Optional[float] = None
):
    """
    Atualiza os principais campos de uma operação existente no banco.
    Agora também marca data_fechamento e campos expandidos.
    """
    try:
        _ensure_dados_dir()
        con = duckdb.connect(db_path)
        _ensure_schema(con)

        updates = []
        params: List[Any] = []

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
            log_event(f"[OPERACAO] Nenhum dado para atualizar (ticket={ticket})", level="warning")
            con.close()
            return

        sql = f"UPDATE operacoes SET {', '.join(updates)} WHERE ticket = ?"
        params.append(ticket)
        con.execute(sql, params)
        con.close()
        log_event(f"[OPERACAO] Operação {ticket} atualizada no banco.", level="info")

    except Exception as e:
        log_event(f"[OPERACAO] Erro ao atualizar operação no DuckDB: {e}. Gravando fallback CSV.", level="error")
        # No fallback, gravamos um "evento de update" com os campos que chegaram
        record = {k: None for k in CAMPOS_TABELA}
        record.update({
            "ticket": ticket,
            "motivo_fechamento": motivo_fechamento,
            "observacao": observacao,
            "preco_fechamento": preco_fechamento,
            "preco_saida": preco_saida,
            "lucro": lucro,
            "resultado": resultado,
            "sinal": sinal,
            "data_fechamento": data_fechamento or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "motivo_saida": motivo_saida,
            "score_reforco": score_reforco,
            "preco_entrada": preco_entrada,
            "modelo_usado": modelo_usado,
        })
        _write_csv_fallback(record, acao="update")
