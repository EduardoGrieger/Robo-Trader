# utils/protecao_loss.py

import os
from datetime import datetime
import duckdb
from utils.debug_logger import log_event

BLOQUEIO_FILE = "dados/bloqueio_loss_diario.txt"

def atingiu_loss_diario(limite_loss_dia):
    """
    Checa se o loss acumulado do dia atingiu o limite configurado.
    Retorna: (atingiu: bool, perda_dia: float)
    """
    try:
        hoje = datetime.now().strftime("%Y-%m-%d")
        con = duckdb.connect("dados/robodados.duckdb")
        res = con.execute("""
            SELECT SUM(CAST(resultado AS DOUBLE)) 
            FROM operacoes
            WHERE resultado IS NOT NULL
              AND DATE(timestamp) = ?
        """, (hoje,)).fetchone()
        perda_dia = float(res[0] or 0)
        con.close()
        if perda_dia <= limite_loss_dia:
            log_event(f"[PROTEÇÃO LOSS] Limite diário de loss atingido: {perda_dia:.2f} (limite: {limite_loss_dia:.2f})", level="warning")
            return True, perda_dia
        log_event(f"[PROTEÇÃO LOSS] Loss diário dentro do limite: {perda_dia:.2f} (limite: {limite_loss_dia:.2f})", level="debug")
        return False, perda_dia
    except Exception as e:
        log_event(f"[PROTEÇÃO LOSS] ERRO ao calcular loss diário: {e}", level="error")
        return False, 0.0

def bloquear_entradas_loss():
    """
    Bloqueia a abertura de novas ordens até o fim do dia.
    """
    hoje = datetime.now().strftime("%Y-%m-%d")
    try:
        with open(BLOQUEIO_FILE, "w") as f:
            f.write(hoje)
        log_event(f"[PROTEÇÃO LOSS] Bloqueio de novas ordens ativado até o fim do dia {hoje}.", level="warning")
    except Exception as e:
        log_event(f"[PROTEÇÃO LOSS] ERRO ao ativar bloqueio de ordens: {e}", level="error")

def esta_bloqueado_loss():
    """
    Retorna True se o robô está bloqueado para novas ordens por loss diário.
    Libera automaticamente ao virar o dia.
    """
    if not os.path.exists(BLOQUEIO_FILE):
        return False
    try:
        with open(BLOQUEIO_FILE, "r") as f:
            data_bloqueio = f.read().strip()
        hoje = datetime.now().strftime("%Y-%m-%d")
        if data_bloqueio == hoje:
            log_event(f"[PROTEÇÃO LOSS] Robô bloqueado para novas ordens pelo restante do dia {hoje}.", level="info")
            return True
        else:
            # Virou o dia, remove o bloqueio
            try:
                os.remove(BLOQUEIO_FILE)
                log_event(f"[PROTEÇÃO LOSS] Bloqueio de loss liberado automaticamente. Novo dia iniciado ({hoje}).", level="info")
            except Exception as e:
                log_event(f"[PROTEÇÃO LOSS] ERRO ao remover arquivo de bloqueio: {e}", level="warning")
            return False
    except Exception as e:
        log_event(f"[PROTEÇÃO LOSS] ERRO ao checar bloqueio: {e}", level="error")
        return False
