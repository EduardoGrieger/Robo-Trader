# utils/db_schema_manager.py

import duckdb
import os
from utils.debug_logger import log_event

def garantir_schema_operacoes(db_path, colunas_necessarias):
    """
    Garante a existência da tabela 'operacoes' e das colunas obrigatórias no banco DuckDB.
    Cria a tabela se não existir, adiciona colunas que faltam e gera logs institucionais.
    """
    try:
        con = duckdb.connect(db_path)
        try:
            # Verifica colunas já existentes
            try:
                result = con.execute("PRAGMA table_info('operacoes')").fetchall()
                colunas_existentes = [col[1] for col in result]
                tabela_existe = len(result) > 0
            except Exception:
                colunas_existentes = []
                tabela_existe = False

            if not tabela_existe:
                log_event("[DB] Tabela 'operacoes' não encontrada. Criando do zero...", level="warning")
                cols_str = ", ".join([f"{col} {tipo}" for col, tipo in colunas_necessarias.items()])
                con.execute(f"CREATE TABLE operacoes ({cols_str})")
                log_event("[DB] Tabela 'operacoes' criada.", level="info")
            else:
                log_event("[DB] Tabela 'operacoes' já existe. Verificando colunas...", level="debug")
                for coluna, tipo in colunas_necessarias.items():
                    if coluna not in colunas_existentes:
                        log_event(f"[DB] Coluna '{coluna}' não encontrada. Adicionando...", level="warning")
                        con.execute(f"ALTER TABLE operacoes ADD COLUMN {coluna} {tipo}")
                        log_event(f"[DB] Coluna '{coluna}' adicionada à tabela 'operacoes'.", level="info")
        finally:
            con.close()
    except Exception as e:
        log_event(f"[DB] Erro ao garantir schema da tabela 'operacoes': {e}", level="error")
