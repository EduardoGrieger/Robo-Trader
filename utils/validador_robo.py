# utils/validador_robo.py
# Validador N+1 robusto:
# - Fase t: registrar_acao(...) grava a decisão do robô como pendente
# - Fase t+1: validar_pendentes(...) encontra o candle N+1, calcula sinal_ideal_n1,
#   compara com sinal_robo e preenche resultado/lucro_pips_n1
# - Correção de timezone (tz-naive), pip_factors por ativo e tolerância por timeframe

import os
import json
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------------
# Config / Utils
# -----------------------------------------------------------------------------
def _log(tag: str, msg: str, level: str = "info"):
    try:
        from utils.debug_logger import log_event
        log_event(tag, msg if level == "info" else f"[{level.upper()}] {msg}")
    except Exception:
        # fallback silencioso
        pass

try:
    from utils.utils import carregar_config
except Exception:
    def carregar_config():
        cfg_path = Path("config.json")
        if cfg_path.exists():
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        return {}

try:
    from utils.sinal_utils import normalizar_sinal
except Exception:
    def normalizar_sinal(s):
        try:
            return int(s)
        except Exception:
            return 0

CAMINHO_CSV = os.path.join("dados", "validacao_decisoes.csv")

# -----------------------------------------------------------------------------
# Helpers de tempo / pip / params
# -----------------------------------------------------------------------------
def _normalize_ts_scalar(ts) -> Optional[pd.Timestamp]:
    """Converte qualquer entrada de tempo para Timestamp tz-naive (remove fuso)."""
    if ts is None:
        return None
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    if t is pd.NaT:
        t = pd.to_datetime(ts, errors="coerce")
    if isinstance(t, pd.Timestamp) and t.tzinfo is not None:
        try:
            t = t.tz_convert(None)
        except Exception:
            try:
                t = t.tz_localize(None)
            except Exception:
                pass
    return t if isinstance(t, pd.Timestamp) else None

def _normalize_ts_series(s: pd.Series) -> pd.Series:
    """Normaliza uma série de tempos para tz-naive."""
    s = pd.to_datetime(s, utc=True, errors="coerce")
    try:
        s = s.dt.tz_convert(None)
    except Exception:
        try:
            s = s.dt.tz_localize(None)
        except Exception:
            pass
    return s

def _timeframe_minutes(tf_str: str) -> int:
    """Converte strings de timeframe típicas ('M5','H1','D1') em minutos aproximados."""
    if not tf_str:
        return 5
    tf_str = str(tf_str).upper()
    try:
        if tf_str.startswith("M"):
            return int(tf_str[1:])
        if tf_str.startswith("H"):
            return int(tf_str[1:]) * 60
        if tf_str.startswith("D"):
            return int(tf_str[1:]) * 1440
    except Exception:
        pass
    return 5

def _params_label(cfg: Dict[str, Any], ativo: str) -> Tuple[int, int, float, int]:
    """
    TP/SL e pip_factor por ativo, além da tolerância (segundos) para casar candle t ~ timestamp do registro.
    """
    lp = cfg.get("label_params", {}) if isinstance(cfg, dict) else {}
    tp = int(lp.get("tp_pips", 40))
    sl = int(lp.get("sl_pips", 20))
    if isinstance(lp.get(ativo, {}), dict):
        tp = int(lp[ativo].get("tp_pips", tp))
        sl = int(lp[ativo].get("sl_pips", sl))

    # pip_factors
    pf_map = cfg.get("pip_factors", {}) or {}
    pf = float(pf_map.get(ativo, 0.0001))
    if pf == 0.0001 and "JPY" in (ativo or "").upper():
        pf = 0.01  # fallback razoável

    # tolerância por timeframe do ativo
    tf_map = cfg.get("timeframes", {}) or {}
    tf = tf_map.get(ativo, "M5")
    tol_sec = max(60, _timeframe_minutes(tf) * 60)  # pelo menos 1 minuto

    return tp, sl, pf, tol_sec

def _achar_indices(df_candles: pd.DataFrame, ts: pd.Timestamp, tol_sec: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Busca índice do candle t e do N+1 usando match exato ou 'nearest' com tolerância.
    """
    idx = pd.DatetimeIndex(df_candles["datahora"])
    try:
        pos_t = idx.get_loc(ts)
    except KeyError:
        # tenta nearest com tolerância
        near = idx.get_indexer([ts], method="nearest")[0]
        if near < 0:
            return None, None
        delta = abs((pd.Timestamp(idx[near]) - pd.Timestamp(ts)).total_seconds())
        if delta > tol_sec:
            return None, None
        pos_t = int(near)
    pos_n1 = pos_t + 1 if (pos_t + 1) < len(idx) else None
    return pos_t, pos_n1

# -----------------------------------------------------------------------------
# CSV de validação
# -----------------------------------------------------------------------------
_DTYPE_MAP = {
    "datahora": "datetime64[ns]",
    "ativo": "object",
    "sinal_robo": "Int64",
    "preco_entrada": "float64",
    "padrao": "object",
    "confianca": "float64",
    "motivo": "object",
    "sinal_ideal_n1": "Int64",
    "resultado": "object",
    "lucro_pips_n1": "float64",
    "pendente": "boolean",
}

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    needed = list(_DTYPE_MAP.keys())
    for c in needed:
        if c not in df.columns:
            df[c] = pd.Series([None] * len(df))
    return df[needed]

def _apply_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # datas
    df["datahora"] = _normalize_ts_series(df["datahora"])
    # numéricos/strings/bool
    for col, dt in _DTYPE_MAP.items():
        if col == "datahora":
            continue
        try:
            if dt == "Int64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dt == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            elif dt == "boolean":
                # pendente como boolean (aceita NA, mas normalizamos depois)
                df[col] = df[col].astype("boolean")
            elif dt == "object":
                df[col] = df[col].astype("object")
        except Exception:
            # última linha de defesa
            pass
    # padrão para pendente: True quando NA na carga (não-rotuladas)
    if "pendente" in df.columns:
        df["pendente"] = df["pendente"].fillna(True).astype("boolean")
    return df

def _read_validacoes() -> pd.DataFrame:
    os.makedirs("dados", exist_ok=True)
    if os.path.exists(CAMINHO_CSV):
        df = pd.read_csv(CAMINHO_CSV)
    else:
        # cria DataFrame vazio já com colunas esperadas
        df = pd.DataFrame(columns=list(_DTYPE_MAP.keys()))
    df = _ensure_columns(df)
    df = _apply_dtypes(df)
    return df

def _write_validacoes(df: pd.DataFrame):
    os.makedirs("dados", exist_ok=True)
    df.to_csv(CAMINHO_CSV, index=False)

# -----------------------------------------------------------------------------
# API Fase t: registrar
# -----------------------------------------------------------------------------
def registrar_acao(
    datahora: Any,
    ativo: str,
    sinal_robo: int,
    preco_entrada: float,
    padrao: Optional[str] = "",
    confianca: Optional[float] = None,
    motivo: Optional[str] = ""
) -> None:
    """Registra a decisão do ciclo t como pendente para validar no t+1."""
    try:
        df = _read_validacoes()
        ts = _normalize_ts_scalar(datahora) or pd.Timestamp.utcnow().tz_localize(None)

        nova = {
            "datahora": ts,
            "ativo": str(ativo),
            "sinal_robo": int(normalizar_sinal(sinal_robo)) if sinal_robo is not None else pd.NA,
            "preco_entrada": float(preco_entrada) if preco_entrada is not None else pd.NA,
            "padrao": padrao if padrao is not None else "",
            "confianca": float(confianca) if confianca is not None else pd.NA,
            "motivo": motivo if motivo is not None else "",
            "sinal_ideal_n1": pd.NA,
            "resultado": pd.NA,
            "lucro_pips_n1": pd.NA,
            "pendente": True,
        }

        # monta df_new, alinha colunas e TIPOS
        df_new = pd.DataFrame([nova])
        df_new = _ensure_columns(df_new)
        # aplica dtypes alvo do CSV
        for col, dt in _DTYPE_MAP.items():
            if col == "datahora":
                df_new["datahora"] = _normalize_ts_series(df_new["datahora"])
            else:
                try:
                    if dt == "Int64":
                        df_new[col] = pd.to_numeric(df_new[col], errors="coerce").astype("Int64")
                    elif dt == "float64":
                        df_new[col] = pd.to_numeric(df_new[col], errors="coerce").astype("float64")
                    elif dt == "boolean":
                        df_new[col] = df_new[col].astype("boolean")
                    elif dt == "object":
                        df_new[col] = df_new[col].astype("object")
                except Exception:
                    pass

        # >>> Fix final do FutureWarning:
        # se o CSV estiver vazio, não concatena — apenas atribui
        if df.empty:
            df = df_new
        else:
            df = pd.concat([df, df_new[df.columns]], ignore_index=True)

        _write_validacoes(df)
        _log("validador", f"[REGISTRO] Ação registrada para {ativo} @ {ts} (sinal={nova['sinal_robo']}, pendente=True)")
    except Exception as e:
        _log("validador", f"[ERRO REGISTRO] {e}", level="error")

# -----------------------------------------------------------------------------
# API Fase t+1: validar
# -----------------------------------------------------------------------------
def validar_pendentes(df_candles: pd.DataFrame, ativo: str) -> int:
    """
    Valida todas as linhas pendentes para 'ativo' usando o candle N+1.
    df_candles precisa conter ['datahora','open','high','low','close'] do mesmo ativo.
    Retorna a quantidade de linhas validadas neste ciclo.
    """
    try:
        if df_candles is None or len(df_candles) == 0:
            _log("validador", "[VALIDAR] df_candles vazio — nada a validar agora.")
            return 0

        # Normaliza datas em tz-naive e ordena
        df_candles = df_candles.copy()
        if "datahora" not in df_candles.columns:
            raise ValueError("df_candles precisa conter a coluna 'datahora'")
        for col in ["open", "high", "low", "close"]:
            if col not in df_candles.columns:
                raise ValueError(f"df_candles precisa conter a coluna '{col}'")

        df_candles["datahora"] = _normalize_ts_series(df_candles["datahora"])
        df_candles.sort_values("datahora", inplace=True, kind="mergesort")
        df_candles.reset_index(drop=True, inplace=True)

        cfg = carregar_config()
        tp_pips, sl_pips, pip_factor, tol_sec = _params_label(cfg, ativo)

        df = _read_validacoes()
        if df.empty:
            _log("validador", "[VALIDAR] CSV de validação vazio.")
            return 0

        mask = (df["pendente"] == True) & (df["ativo"].astype(str).str.upper() == (ativo or "").upper())
        pend = df[mask]
        if pend.empty:
            _log("validador", f"[VALIDAR] Sem pendentes para {ativo}.")
            return 0

        validadas = 0
        for idx in pend.index:
            try:
                ts = _normalize_ts_scalar(df.at[idx, "datahora"])
                if ts is None:
                    continue

                pos_t, pos_n1 = _achar_indices(df_candles, ts, tol_sec=tol_sec)
                if pos_t is None or pos_n1 is None:
                    # Ainda não é possível validar (sem N+1 ou timestamp não casou dentro da tolerância)
                    continue

                c_t  = df_candles.iloc[pos_t]
                c_n1 = df_candles.iloc[pos_n1]

                entry = df.at[idx, "preco_entrada"]
                if entry is None or pd.isna(entry):
                    entry = float(c_t["close"])

                # Avaliação intrabar no N+1 (TP/SL)
                take_buy  = entry + tp_pips * pip_factor
                stop_buy  = entry - sl_pips * pip_factor
                take_sell = entry - tp_pips * pip_factor
                stop_sell = entry + sl_pips * pip_factor

                high  = float(c_n1["high"])
                low   = float(c_n1["low"])
                close = float(c_n1["close"])
                open_ = float(c_n1["open"])

                hit_tp_buy  = high >= take_buy
                hit_sl_buy  = low  <= stop_buy
                hit_tp_sell = low  <= take_sell
                hit_sl_sell = high >= stop_sell

                ideal = 0
                if hit_tp_buy and not hit_sl_buy:
                    ideal = 1
                elif hit_sl_buy and not hit_tp_buy:
                    ideal = -1
                elif hit_tp_sell and not hit_sl_sell:
                    ideal = -1
                elif hit_sl_sell and not hit_tp_sell:
                    ideal = 1
                elif (hit_tp_buy and hit_sl_buy) or (hit_tp_sell and hit_sl_sell):
                    # desempate por proximidade dos níveis ao preço de abertura do N+1
                    dist_buy_tp  = abs(take_buy  - open_)
                    dist_buy_sl  = abs(stop_buy  - open_)
                    dist_sell_tp = abs(take_sell - open_)
                    dist_sell_sl = abs(stop_sell - open_)
                    ideal = 1 if min(dist_buy_tp,  dist_sell_sl) < min(dist_sell_tp, dist_buy_sl) else -1
                else:
                    # fallback por direção do fechamento vs entrada (metade do TP)
                    delta = close - entry
                    if abs(delta) >= (tp_pips / 2.0) * pip_factor:
                        ideal = 1 if delta > 0 else -1
                    else:
                        ideal = 0

                # Resultado e lucro
                robo = int(normalizar_sinal(df.at[idx, "sinal_robo"]))
                if ideal == 0 and robo == 0:
                    resultado = "neutro"
                elif robo == ideal and robo != 0:
                    resultado = "acertou"
                elif robo != 0 and ideal != 0 and robo != ideal:
                    resultado = "errou"
                else:
                    resultado = "errou" if robo != 0 else "neutro"

                # lucro_pips seguindo o ROBÔ até o fechamento do N+1
                if robo == 1:
                    lucro_pips = (close - entry) / pip_factor
                elif robo == -1:
                    lucro_pips = (entry - close) / pip_factor
                else:
                    lucro_pips = 0.0

                # Persistência
                df.at[idx, "sinal_ideal_n1"] = int(ideal)
                df.at[idx, "resultado"] = resultado
                df.at[idx, "lucro_pips_n1"] = round(float(lucro_pips), 2)
                df.at[idx, "pendente"] = False

                validadas += 1
                _log("validador",
                     f"[N+1] {ativo} @ {ts} ideal={ideal} robo={robo} => {resultado} lucro_pips_n1={round(float(lucro_pips),2)}")

            except Exception as e:
                _log("validador", f"[ERRO VALIDAR] {e}", level="error")

        if validadas > 0:
            _write_validacoes(df)
        else:
            _log("validador", f"[VALIDAR] Nenhuma linha apta a validar (aguardando N+1 ou timestamp fora da tolerância) para {ativo}.")

        return validadas

    except Exception as e:
        _log("validador", f"[ERRO VALIDAR] {e}", level="error")
        return 0
