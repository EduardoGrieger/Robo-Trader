# utils/utils.py
import json
import time
from datetime import datetime, timedelta
import os
import pandas as pd
import joblib
import numpy as np
from utils.debug_logger import log_event

# =========================
# Helpers internos de arquivo
# =========================
def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _safe_write_text(path: str, text: str) -> None:
    """Escrita atômica para TXT/CSV de pequeno porte."""
    _ensure_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)

def _safe_write_df_csv(df: pd.DataFrame, path: str) -> None:
    """Escrita atômica de DataFrame em CSV."""
    _ensure_dir(path)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

# =========================
# Config
# =========================
_CONFIG_CACHE = None
_CONFIG_PATH = None

def carregar_config(caminho: str = "config.json", force_reload: bool = False) -> dict:
    """
    Carrega o arquivo de configuração e retorna como dict.
    - Cacheia em memória para chamadas repetidas no ciclo.
    - Em erro, retorna {} e loga o problema.
    - force_reload=True força releitura do arquivo.
    """
    global _CONFIG_CACHE, _CONFIG_PATH
    try:
        if (not force_reload) and (_CONFIG_CACHE is not None) and (_CONFIG_PATH == caminho):
            return _CONFIG_CACHE

        with open(caminho, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            log_event(f"[UTILS] config.json não é um objeto (dict). Usando vazio.", level="warning")
            cfg = {}
        _CONFIG_CACHE, _CONFIG_PATH = cfg, caminho
        return cfg
    except Exception as e:
        log_event(f"[UTILS] ERRO ao carregar config ({caminho}): {e}", level="error")
        _CONFIG_CACHE, _CONFIG_PATH = {}, caminho
        return {}

# =========================
# Sincronização com o candle/ciclo
# =========================
def aguardar_inicio_novo_candle(timeframe_min: int = 1, verbose: bool = False) -> None:
    """
    Aguarda até o início do próximo candle do timeframe (em minutos).
    Ex.: timeframe_min=5 sincroniza nos minutos múltiplos de 5 (00,05,10,...).
    """
    agora = datetime.now().replace(microsecond=0)
    if timeframe_min > 1:
        minuto_atual = agora.minute
        proximo_minuto = ((minuto_atual // timeframe_min) + 1) * timeframe_min
        base = agora.replace(second=0)
        if proximo_minuto >= 60:
            alvo = base.replace(minute=0) + timedelta(hours=1)
        else:
            alvo = base.replace(minute=proximo_minuto)
    else:
        alvo = agora.replace(second=0) + timedelta(minutes=1)

    if verbose:
        log_event(f"[SYNC] Aguardando até: {alvo.strftime('%H:%M:%S')}", level="debug")

    # Espera com pequenos sleeps para reduzir drift
    while True:
        restante = (alvo - datetime.now()).total_seconds()
        if restante <= 0:
            break
        time.sleep(min(0.5, max(0.05, restante / 10.0)))

def aguardar_proximo_ciclo(intervalo_minutos: int = 5, verbose: bool = False) -> None:
    """
    Aguarda até o próximo ciclo de execução baseado no intervalo (minutos).
    Ex.: intervalo=5 executa em 00,05,10,...
    """
    agora = datetime.now().replace(microsecond=0)
    base = agora.replace(second=0)
    minutos_ate = intervalo_minutos - (agora.minute % intervalo_minutos)
    if minutos_ate == 0:
        minutos_ate = intervalo_minutos
    alvo = base + timedelta(minutes=minutos_ate)

    if verbose:
        log_event(f"[SYNC] Aguardando até: {alvo.strftime('%H:%M:%S')}", level="debug")

    while True:
        restante = (alvo - datetime.now()).total_seconds()
        if restante <= 0:
            break
        time.sleep(min(0.5, max(0.05, restante / 10.0)))

# =========================
# Calendário/Horários
# =========================
def eh_dia_util(timestamp: datetime | None = None) -> bool:
    """Dia útil *para Forex*: considera domingo à noite como dia útil
    e sexta após o fechamento como NÃO útil. Usa janela global em UTC.
    """
    # Mantemos assinatura para compatibilidade; delegamos para mercado_esta_aberto().
    return mercado_esta_aberto()

def mercado_esta_aberto() -> bool:
    """
    Janela típica de Forex em UTC (com override opcional via config.json):
      - Abre domingo às 21:00 UTC (default)
      - Fecha sexta às 21:00 UTC (default)
    Algumas corretoras variam (21:00/22:00 UTC, mudanças por DST). Você pode
    ajustar em config.json -> {"forex_open_utc": "21:00", "forex_close_utc": "21:00"}.
    """
    from datetime import datetime
    cfg = carregar_config()
    open_str = str(cfg.get("forex_open_utc", "21:00"))
    close_str = str(cfg.get("forex_close_utc", "21:00"))

    def _hhmm(s: str) -> tuple[int, int]:
        try:
            h, m = s.split(":", 1)
            return int(h), int(m)
        except Exception:
            return 21, 0

    open_h, open_m = _hhmm(open_str)
    close_h, close_m = _hhmm(close_str)

    now = datetime.utcnow()
    wd = now.weekday()  # 0=Seg ... 6=Dom
    hhmm = (now.hour, now.minute)
    open_hhmm = (open_h, open_m)
    close_hhmm = (close_h, close_m)

    if wd < 4:  # Seg–Qui: aberto
        return True
    if wd == 4:  # Sexta: aberto até o horário de fechamento
        return hhmm < close_hhmm
    if wd == 6:  # Domingo: abre a partir do horário de abertura
        return hhmm >= open_hhmm
    # Sábado
    return False

# =========================
# Logs auxiliares
# =========================
def salvar_log(obj, caminho: str = "logs/log.txt") -> None:
    """
    Se 'obj' for DataFrame -> grava CSV (escrita atômica).
    Caso contrário -> anexa linha de texto (com timestamp).
    """
    try:
        if isinstance(obj, pd.DataFrame):
            _safe_write_df_csv(obj, caminho)
        else:
            agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            linha = f"[{agora}] {str(obj)}\n"
            # Para TXT/LOG, manter append, sem atômico para não perder histórico
            _ensure_dir(caminho)
            with open(caminho, "a", encoding="utf-8") as f:
                f.write(linha)
    except Exception as e:
        log_event(f"[UTILS] ERRO ao salvar log ({caminho}): {e}", level="error")

# =========================
# Preparação de features
# =========================
def preparar_para_previsao(df: pd.DataFrame, features_salvas_path: str) -> pd.DataFrame:
    """
    Prepara DataFrame de previsão para o modelo:
    - Alinha/ordena colunas conforme features salvas
    - Converte bool -> int (0/1), trata NaN/Inf com 0
    - Loga colunas faltantes/excedentes para auditoria
    """
    try:
        features_treinadas = joblib.load(features_salvas_path)
        if not isinstance(features_treinadas, (list, tuple)):
            raise ValueError("features_treinadas inválidas (esperado list/tuple)")

        # Somente numéricas/booleanas
        df_num = df.select_dtypes(include=[np.number, "bool"]).copy()

        # Converte bool para int
        for col in df_num.columns[df_num.dtypes == "bool"]:
            df_num[col] = df_num[col].astype(np.int8)

        # Alinha e preenche ausências com 0
        faltantes = [c for c in features_treinadas if c not in df_num.columns]
        extras = [c for c in df_num.columns if c not in features_treinadas]

        df_alinhado = df_num.reindex(columns=features_treinadas, fill_value=0)

        # Trata NaN/Inf
        df_alinhado = df_alinhado.replace([np.inf, -np.inf], 0).fillna(0)

        if faltantes:
            log_event(f"[PREPARAR_PREVISAO] Faltando {len(faltantes)} colunas do treino: {faltantes[:20]}{'...' if len(faltantes)>20 else ''}", level="warning")
        if extras:
            log_event(f"[PREPARAR_PREVISAO] {len(extras)} colunas não usadas pelo modelo (ignoradas): {extras[:20]}{'...' if len(extras)>20 else ''}", level="debug")

        log_event(f"[PREPARAR_PREVISAO] Total features: {len(features_treinadas)}", level="debug")
        return df_alinhado

    except Exception as e:
        log_event(f"[UTILS] ERRO ao preparar para previsão: {e}", level="error")
        raise
