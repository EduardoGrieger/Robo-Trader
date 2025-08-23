# utils/eh_feriado.py

import os
import pandas as pd
from datetime import datetime
from utils.debug_logger import log_event

FERIADOS_CSV = "dados/feriados.csv"

def eh_feriado():
    if not os.path.exists(FERIADOS_CSV):
        return False
    try:
        df = pd.read_csv(FERIADOS_CSV)
        hoje = datetime.now().strftime("%Y-%m-%d")
        cols = [c.lower() for c in df.columns]
        if 'data' not in cols:
            log_event(f"[FERIADO] CSV de feriados existe mas não tem coluna 'data'.", level="warning")
            return False
        col_data = df.columns[cols.index('data')]
        datas = df[col_data].astype(str).str.strip()
        if hoje in datas.values:
            log_event(f"[FERIADO] Hoje ({hoje}) é feriado! Robô pausado.", level="info")
            return True
        return False
    except Exception as e:
        log_event(f"[FERIADO] Falha ao checar feriado: {e}", level="warning")
        return False
