# features/gerar_features.py
# --------------------------------------------
# Geração de features robusta (sem look-ahead),
# HMM/MT5/Telegram opcionais, e label TP/SL com
# prioridade intra-candle coerente com a execução.
# --------------------------------------------

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")
except Exception:
    pass

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

# ---- Telegram (opcional: pacote ou arquivo solto), sem warning do Pylance ----
def _carregar_enviar_telegram():
    # 1) pacote em comunicacao/
    try:
        from comunicacao.telegram_alertas import enviar_telegram as _send  # type: ignore
        return _send
    except Exception:
        pass
    # 2) módulo solto telegram_alertas.py via importlib
    try:
        import importlib
        mod = importlib.import_module("telegram_alertas")
        return getattr(mod, "enviar_telegram")
    except Exception:
        pass
    # 3) stub silencioso
    def _stub(*args, **kwargs):
        return False
    return _stub

enviar_telegram = _carregar_enviar_telegram()

# ---- HMM (opcional) ----
try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
    HAS_HMM = True
except Exception:
    GaussianHMM = None  # type: ignore
    HAS_HMM = False

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils.utils import carregar_config
from utils.sinal_utils import normalizar_sinal
from utils.debug_logger import log_event


#############################
# Funções de features
#############################

def calcular_macd(df):
    df = df.copy()
    if len(df) < 2:
        df["macd"] = 0
        df["macd_signal"] = 0
        df["macd_hist"] = 0
        return df
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    log_event("[features] MACD calculado")
    return df

def calcular_ema_sma(df):
    df = df.copy()
    if len(df) < 2:
        df["ema_21"] = 0
        df["ema_200"] = 0
        df["ema_distance"] = 0
        df["sma_20"] = 0
        df["sma_50"] = 0
        return df
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["ema_distance"] = df["ema_21"] - df["ema_200"]
    df["sma_20"] = df["close"].rolling(20, min_periods=1).mean()
    df["sma_50"] = df["close"].rolling(50, min_periods=1).mean()
    log_event("[features] EMA/SMA calculadas")
    return df

def calcular_atr(df):
    df = df.copy()
    if len(df) < 2:
        df["atr_14"] = 0
        df["atr_pct"] = 0
        df["tr"] = 0
        return df
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(window=14, min_periods=1).mean()
    df["atr_pct"] = df["atr_14"] / (df["close"] + 1e-9)
    df["tr"] = tr
    log_event("[features] ATR calculado")
    return df

def calcular_adx(df):
    df = df.copy()
    if len(df) < 2:
        df["plus_di"] = 0
        df["minus_di"] = 0
        df["adx_14"] = 0
        df["adx_strength"] = 0
        return df
    tr = df["tr"] if "tr" in df.columns else (df["high"] - df["low"]).rolling(14, min_periods=1).sum()
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr14 = tr.rolling(window=14, min_periods=1).sum()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window=14, min_periods=1).sum() / (tr14 + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window=14, min_periods=1).sum() / (tr14 + 1e-9)
    adx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)).rolling(window=14, min_periods=1).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx_14"] = adx
    df["adx_strength"] = (adx > 25).astype(int)
    log_event("[features] ADX calculado")
    return df

def calcular_cci(df):
    df = df.copy()
    if len(df) < 2:
        df["cci"] = 0
        df["cci_divergence"] = 0
        return df
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cci = (tp - tp.rolling(20, min_periods=1).mean()) / (0.015 * tp.rolling(20, min_periods=1).std())
    df["cci"] = cci
    df["cci_divergence"] = (cci.diff() > 0).astype(int)
    log_event("[features] CCI calculado")
    return df

def calcular_stochastic(df):
    df = df.copy()
    if len(df) < 2:
        df["stoch_k"] = 0
        df["stoch_d"] = 0
        df["stoch_cross"] = 0
        return df
    lowest_low = df["low"].rolling(window=14, min_periods=1).min()
    highest_high = df["high"].rolling(window=14, min_periods=1).max()
    df["stoch_k"] = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(window=3, min_periods=1).mean()
    df["stoch_cross"] = (df["stoch_k"] > df["stoch_d"]).astype(int)
    log_event("[features] Stochastic calculado")
    return df

def calcular_price_action(df):
    df = df.copy()
    if len(df) < 2:
        df["bullish_engulfing"] = 0
        df["bearish_engulfing"] = 0
        df["body_position"] = 0
        df["upper_shadow_ratio"] = 0
        df["lower_shadow_ratio"] = 0
        df["retorno_condicional"] = 0
        return df
    df["bullish_engulfing"] = ((df["close"].shift(1) < df["open"].shift(1)) &
                               (df["close"] > df["open"]) &
                               (df["close"] > df["open"].shift(1)) &
                               (df["open"] < df["close"].shift(1))).astype(int)
    df["bearish_engulfing"] = ((df["close"].shift(1) > df["open"].shift(1)) &
                               (df["close"] < df["open"]) &
                               (df["close"] < df["open"].shift(1)) &
                               (df["open"] > df["close"].shift(1))).astype(int)
    df["body_position"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
    df["upper_shadow_ratio"] = (df["high"] - df[["close", "open"]].max(axis=1)) / (df["high"] - df["low"] + 1e-9)
    df["lower_shadow_ratio"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-9)
    df["retorno_condicional"] = df["close"].pct_change().where(
        (df["high"] - df["low"]) > (df["high"] - df["low"]).rolling(20, min_periods=1).mean(), 0
    )
    log_event("[features] Price Action calculado")
    return df

def _to_utc_cols(df):
    """
    Constrói colunas UTC e epoch a partir de 'data_hora' OU 'timestamp' OU 'time'.
    Garante Series alinhada ao índice (nunca escalar) e respeita
    config['timezone_servidor_offset_horas'].
    """
    import pandas as pd, numpy as np
    try:
        cfg = carregar_config()
    except Exception:
        cfg = {}

    # 1) Escolhe base de tempo sempre como SERIES (evita escalar que quebra .dt)
    if "data_hora" in df.columns:
        base = pd.to_datetime(df["data_hora"], errors="coerce")
    elif "timestamp" in df.columns:
        base = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "time" in df.columns:
        base = pd.to_datetime(df["time"], errors="coerce")
    else:
        # fallback: Series com 'agora' no mesmo índice
        base = pd.Series(pd.Timestamp.utcnow(), index=df.index, dtype="datetime64[ns]")

    # 2) Aplica offset do servidor para chegar em UTC
    try:
        offset_h = float(cfg.get("timezone_servidor_offset_horas", 0.0))
    except Exception:
        offset_h = 0.0
    # Ex.: offset_h=-3 (servidor UTC-3) -> UTC = local - (-3h) = local + 3h
    base_utc = base - pd.to_timedelta(offset_h, unit="h")

    # 3) Colunas finais (com .dt seguro porque é Series)
    df["data_hora_utc"]   = base_utc
    df["timestamp_utc"]   = base_utc.dt.strftime("%Y-%m-%d %H:%M:%S")

    # Epoch (segundos); NaT -> NaN
    try:
        ns   = base_utc.view("int64")              # datetime64[ns] -> int (ns)
        secs = (ns // 1_000_000_000)
        secs = secs.where(base_utc.notna(), np.nan)
    except Exception:
        secs = pd.Series(np.nan, index=df.index)
    df["timestamp_epoch"] = secs.astype(float)

    return df

def calcular_contexto_mercado(df):
    df = df.copy()
    if len(df) < 2:
        df["timestamp"] = ""
        df["data_hora"] = pd.to_datetime("now")
        df["data_hora_utc"] = pd.to_datetime("now", utc=True)
        df["timestamp_utc"] = pd.to_datetime("now", utc=True).strftime("%Y-%m-%d %H:%M:%S")
        df["timestamp_epoch"] = np.nan
        df["hour_sin"] = 0
        df["hour_cos"] = 0
        df["vol_ratio"] = 0
        df["tick_vol_zscore"] = 0
        return df
    if "time" in df.columns:
        df["timestamp"] = df["time"]
        df["data_hora"] = pd.to_datetime(df["time"])
    else:
        df["data_hora"] = pd.to_datetime("now")

    # >>> NOVO: garantir colunas UTC e epoch, e usar UTC para seno/cosseno da hora
    df = _to_utc_cols(df)
    hora = df["data_hora_utc"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hora / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hora / 24)

    df["vol_ratio"] = (df["high"] - df["low"]) / (df["close"].rolling(20, min_periods=1).std() + 1e-9)
    if "tick_volume" in df.columns:
        z1h = (df["tick_volume"].rolling(12, min_periods=1).sum() - df["tick_volume"].rolling(288, min_periods=1).mean()) / \
              (df["tick_volume"].rolling(288, min_periods=1).std() + 1e-9)
        df["tick_vol_zscore"] = z1h
    else:
        df["tick_vol_zscore"] = 0
    log_event("[features] Contexto de mercado (UTC-aware) calculado")
    return df

def calcular_sinais_derivados(df):
    df = df.copy()
    if len(df) < 2:
        df["breakout_adaptive"] = 0
        df["squeeze"] = 0
        df["rsi"] = 0
        df["rsi_slope"] = 0
        df["bb_high"] = 0
        df["bb_low"] = 0
        return df

    # Breakout adaptativo
    df["breakout_adaptive"] = ((df["high"] > df["high"].shift(1)) &
                               (df["atr_14"] > df["atr_14"].rolling(20, min_periods=1).mean())).astype(int)

    # Bollinger & squeeze (sem look-ahead)
    rolling_mean = df["close"].rolling(20, min_periods=1).mean()
    rolling_std  = df["close"].rolling(20, min_periods=1).std()
    bb_width = 4 * rolling_std / (rolling_mean + 1e-9)

    q_roll = bb_width.rolling(200, min_periods=50).quantile(0.2)
    q_exp  = bb_width.expanding(min_periods=50).quantile(0.2)
    thresh = q_roll.combine_first(q_exp).shift(1)  # evita olhar o candle atual
    df["squeeze"] = (bb_width < thresh.fillna(bb_width)).astype(int)

    # RSI + slope
    window = 14
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window=window, min_periods=1).mean()
    avg_loss = down.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_slope"] = df["rsi"].diff()

    df["bb_high"] = rolling_mean + 2 * rolling_std
    df["bb_low"]  = rolling_mean - 2 * rolling_std
    log_event("[features] Sinais derivados calculados (squeeze sem look-ahead)")
    return df

def calcular_meta_features_treino(df):
    """
    Gera as MESMAS meta-features usadas no treino/inferência:
      - volatility_20, volatility_50 (std de retornos)
      - trend_strength (|slope| / std)
      - range_ratio ((high-low)/sma20(close))
    Sem look-ahead; tolerante a séries curtas.
    """
    df = df.copy()
    ok_close = "close" in df.columns
    ok_h = "high" in df.columns
    ok_l = "low" in df.columns

    try:
        if ok_close:
            ret = df["close"].pct_change()
            df["volatility_20"] = ret.rolling(20).std()
            df["volatility_50"] = ret.rolling(50).std()
            df["trend_strength"] = df["close"].rolling(20).apply(
                lambda x: (abs(np.polyfit(range(len(x)), x, 1)[0]) / (np.std(x) + 1e-12)) if np.std(x) > 0 else 0,
                raw=False
            )
        if ok_close and ok_h and ok_l:
            sma20 = df["close"].rolling(20).mean()
            sma20 = sma20.replace(0, np.nan)
            df["range_ratio"] = (df["high"] - df["low"]) / sma20
            df["range_ratio"] = df["range_ratio"].fillna(0.0)
        log_event("[features] Meta-features (vol20, vol50, trend_strength, range_ratio) geradas")
    except Exception as e:
        log_event(f"[features] Falha ao gerar meta-features: {e}", level="warning")
        # Ainda assim, garante existência das colunas
        for c in ("volatility_20", "volatility_50", "trend_strength", "range_ratio"):
            if c not in df.columns:
                df[c] = 0.0
    return df

def calcular_regime_mercado(df, n_clusters=3):
    if len(df) < 2:
        df = df.copy()
        df["regime_kmeans"] = 0
        return df
    regime_features = df[["atr_14", "vol_ratio"]].fillna(0)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["regime_kmeans"] = model.fit_predict(regime_features)
    log_event(f"[institucional] Regime mercado (KMeans, clusters={n_clusters}) calculado.")
    return df

def calcular_regime_hmm(df, n_states=3):
    if not HAS_HMM or GaussianHMM is None:
        log_event("[HMM] hmmlearn não disponível. Usando KMeans como fallback.", level="warning")
        df = df.copy()
        df["regime_hmm"] = df.get("regime_kmeans", -1)
        return df

    cols = ["atr_14", "vol_ratio", "retorno_condicional"]

    # Pré-cheques
    if len(df) < 2 or not all(col in df.columns for col in cols):
        log_event("[HMM] Colunas ausentes ou DF curto. Usando KMeans como fallback.", level="warning")
        df = df.copy()
        df["regime_hmm"] = df.get("regime_kmeans", -1)
        return df

    # Matriz com diferenças (mitiga não-estacionaridade) e limpeza
    X = df[cols].diff().dropna().values
    if X.size == 0 or X.shape[0] < 50:
        log_event(f"[HMM] Amostra insuficiente para fit (n={X.shape[0]}). Usando KMeans.", level="warning")
        df = df.copy()
        df["regime_hmm"] = df.get("regime_kmeans", -1)
        return df

    X = np.asarray(X, dtype=float)
    X = X[~np.isnan(X).any(axis=1)]
    X = X[~np.isinf(X).any(axis=1)]
    if X.shape[0] < 50:
        log_event(f"[HMM] Amostra útil < 50 após limpeza (n={X.shape[0]}). Usando KMeans.", level="warning")
        df = df.copy()
        df["regime_hmm"] = df.get("regime_kmeans", -1)
        return df

    try:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        model = GaussianHMM(
            n_components=int(n_states),
            covariance_type="diag",
            n_iter=200,
            tol=1e-2,
            random_state=42,
            verbose=False,
            init_params="stmc",
            params="stmc",
        )
        model.fit(X_std)
        regimes = model.predict(X_std)
        pad = len(df) - len(regimes)
        regimes_full = np.concatenate([[regimes[0]] * pad, regimes]) if pad > 0 else regimes
        df = df.copy()
        df["regime_hmm"] = regimes_full[:len(df)]
        log_event(f"[institucional] Regime mercado (HMM, states={n_states}) calculado.")
    except Exception as e:
        log_event(f"[HMM] Erro: {e}. Usando KMeans.", level="warning")
        df = df.copy()
        df["regime_hmm"] = df.get("regime_kmeans", -1)

    return df

def calcular_microestrutura_proxy(df):
    df = df.copy()
    if len(df) < 2:
        df["micro_tick_zscore"] = 0
        df["micro_vol_burst"] = 0
        df["micro_spike"] = 0
        return df
    if "tick_volume" in df.columns:
        rolling = df["tick_volume"].rolling(288, min_periods=1)
        zscore = (df["tick_volume"] - rolling.mean()) / (rolling.std() + 1e-9)
        df["micro_tick_zscore"] = zscore
    else:
        df["micro_tick_zscore"] = 0
    vol_burst = df["close"].rolling(20, min_periods=1).std() / (df["close"].rolling(100, min_periods=1).std() + 1e-9)
    df["micro_vol_burst"] = vol_burst
    df["micro_spike"] = (df["micro_tick_zscore"] > 2).astype(int)
    log_event(f"[institucional] Microestrutura proxies calculados")
    return df

def calcular_anomalias(df):
    df = df.copy()
    if len(df) < 2:
        df["rolling_kurtosis"] = 0
        df["rolling_skewness"] = 0
        df["gap"] = 0
        df["range_outlier"] = 0
        return df
    df["rolling_kurtosis"] = df["close"].rolling(50, min_periods=1).kurt()
    df["rolling_skewness"] = df["close"].rolling(50, min_periods=1).skew()
    df["gap"] = df["open"] - df["close"].shift(1)
    df["range_outlier"] = ((df["high"] - df["low"]) >
                           (df["high"] - df["low"]).rolling(100, min_periods=1).mean() +
                           2 * (df["high"] - df["low"]).rolling(100, min_periods=1).std()).astype(int)
    log_event("[institucional] Anomalias calculadas")
    return df

def calcular_sharpe_rolling(df, window=50):
    df = df.copy()
    if len(df) < 2:
        df["rolling_sharpe"] = 0
        return df
    returns = df["close"].pct_change()
    rolling_mean = returns.rolling(window, min_periods=1).mean()
    rolling_std = returns.rolling(window, min_periods=1).std()
    df["rolling_sharpe"] = (rolling_mean / (rolling_std + 1e-9)).fillna(0)
    log_event("[institucional] Sharpe rolling calculado")
    return df

def calcular_autocorrelacao(df, window=50, lag=1):
    df = df.copy()
    if len(df) < 2:
        df["rolling_autocorr"] = 0
        return df
    try:
        df["rolling_autocorr"] = df["close"].rolling(window, min_periods=1)\
            .apply(lambda x: pd.Series(x).autocorr(lag=lag))
    except Exception as e:
        log_event(f"[institucional] Falha autocorrelação: {e}", level="warning")
        df["rolling_autocorr"] = 0
    log_event("[institucional] Autocorrelação rolling calculada")
    return df

def gerar_sinal_tp_sl_realista(
    df,
    tp_pips=30,
    sl_pips=30,
    janela=10,
    pip_factor=0.0001,
    prioridade_intracandle: str = "SL"
):
    """
    Gera label: 1=TP antes do SL, -1=SL antes do TP, 0=nenhum em 'janela'.
    Quando TP e SL ocorrem no MESMO candle:
      - "SL"   -> escolhe SL (conservador)
      - "TP"   -> escolhe TP (otimista)
      - "ordem"-> decide pelo alvo mais próximo do OPEN desse candle
    """
    sinais = []
    if len(df) < janela + 1:
        df = df.copy()
        df["sinal"] = [0] * len(df)
        return df

    for i in range(len(df) - janela):
        preco = float(df.loc[i, "close"])
        tp = preco + tp_pips * pip_factor
        sl = preco - sl_pips * pip_factor

        sinal = 0
        for j in range(i + 1, i + 1 + janela):
            hi = float(df.loc[j, "high"])
            lo = float(df.loc[j, "low"])
            if hi >= tp and lo <= sl:
                if prioridade_intracandle.upper() == "TP":
                    sinal = 1
                elif prioridade_intracandle.lower() == "ordem":
                    o = float(df.loc[j, "open"]) if "open" in df.columns else preco
                    d_tp = abs(tp - o); d_sl = abs(o - sl)
                    sinal = 1 if d_tp < d_sl else -1
                else:
                    sinal = -1
                break
            if lo <= sl:
                sinal = -1; break
            if hi >= tp:
                sinal = 1; break

        sinais.append(normalizar_sinal(sinal))

    sinais += [0] * janela
    df = df.copy()
    df["sinal"] = sinais

    log_event(f"[label] Label por TP/SL (tp={tp_pips}, sl={sl_pips}, janela={janela}, prio={prioridade_intracandle})")
    dist = pd.Series(sinais).value_counts(normalize=True)
    if not dist.empty and dist.max() > 0.7:
        log_event(f"⚠️ Desbalanceamento forte do label: {dist.to_dict()}", level="warning")
    return df


#############################
# Auditoria e pós-processo
#############################

def pre_processar_final(df):
    df = df.copy()
    # Preenche NaN numéricos por média (simples/seguro para online)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    essenciais = [
        "close", "rsi", "sma_20", "sma_50", "bb_high", "bb_low", "sinal",
        "macd", "macd_signal", "macd_hist", "ema_21", "ema_200", "ema_distance",
        "atr_14", "adx_14", "cci", "stoch_k", "stoch_d",
        # meta-features do treino/inferência
        "volatility_20", "volatility_50", "trend_strength", "range_ratio",
        # novo: epoch ajuda na telemetria e é numérico
        "timestamp_epoch"
    ]

    # Remove colunas “mortas”
    to_drop = []
    for col in df.columns:
        try:
            if col not in essenciais and df[col].nunique(dropna=False) <= 1:
                to_drop.append(col)
        except Exception:
            pass
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")

    # Blinda essenciais
    for col in essenciais:
        if col not in df.columns or df[col].isnull().all():
            df[col] = 0

    if df[essenciais].isnull().all().any():
        log_event("❌ Feature essencial totalmente NaN!", level="error")
    return df

def _assegurar_timestamp(df):
    """
    Garante a existência da coluna 'timestamp' para consumo do main_loop/validador.
    Respeita 'time' e 'data_hora' se existirem; adiciona 'timestamp_utc' e 'timestamp_epoch'.
    """
    df = df.copy()
    if "timestamp" not in df.columns or df["timestamp"].isnull().all():
        if "time" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                df["timestamp"] = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")
        elif "data_hora" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["data_hora"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                df["timestamp"] = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")
        else:
            df["timestamp"] = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")

    # Garante UTC/epoch mesmo se chamado isoladamente
    if "data_hora_utc" not in df.columns or "timestamp_epoch" not in df.columns or "timestamp_utc" not in df.columns:
        df = _to_utc_cols(df)
    return df

def auditar_features_ultima_linha(df, ativo=""):
    """
    Audita a ÚLTIMA linha (linha do sinal). Loga:
    - colunas essenciais ausentes
    - NaNs/inf na última linha
    - valores bizarros (ex.: inf)
    """
    if df is None or df.empty:
        log_event(f"[sanity] DF vazio após features para {ativo}.", level="warning")
        return

    essenciais = [
        "close", "rsi", "sma_20", "sma_50", "bb_high", "bb_low", "sinal",
        "macd", "macd_signal", "macd_hist", "ema_21", "ema_200",
        "atr_14", "adx_14", "cci", "stoch_k", "stoch_d",
        # meta-features do treino/inferência
        "volatility_20", "volatility_50", "trend_strength", "range_ratio",
        # novo: epoch numérico para depuração de tempo
        "timestamp_epoch"
    ]

    ausentes = [c for c in essenciais if c not in df.columns]
    ultima = df.iloc[-1].copy()

    nans = [c for c, v in ultima.items() if pd.isna(v)]

    def _is_inf_val(v):
        try:
            if isinstance(v, (int, float, np.floating, np.integer, np.number)):
                return bool(np.isinf(v))
            return False
        except Exception:
            return False

    infs = [c for c, v in ultima.items() if _is_inf_val(v)]

    resumo = {
        "cols_totais": len(df.columns),
        "nans_ultima_linha": len(nans),
        "infs_ultima_linha": len(infs),
        "essenciais_ausentes": ausentes
    }
    if ausentes or nans or infs:
        log_event(f"[sanity:{ativo}] Problemas na última linha: {resumo}", level="warning")
    else:
        log_event(f"[sanity:{ativo}] Última linha OK ({len(df.columns)} cols).", level="info")


#############################
# Orquestrador institucional
#############################

def calcular_features(candles, config, ativo=""):
    try:
        df = candles.copy()
        log_event(f"[features] Iniciando cálculo para {ativo} ({df.shape[0]} linhas iniciais)")

        df = calcular_macd(df)
        df = calcular_ema_sma(df)
        df = calcular_atr(df)
        df = calcular_adx(df)
        df = calcular_cci(df)
        df = calcular_stochastic(df)
        df = calcular_price_action(df)
        df = calcular_contexto_mercado(df)
        df = calcular_sinais_derivados(df)

        # === META-FEATURES que precisam existir para bater com o treino/inferência ===
        df = calcular_meta_features_treino(df)

        n_clusters = config.get("n_clusters_regime", 3)
        n_states = config.get("n_states_hmm", 3)
        if all(col in df.columns for col in ["atr_14", "vol_ratio"]):
            df = calcular_regime_mercado(df, n_clusters=n_clusters)
        if all(col in df.columns for col in ["atr_14", "vol_ratio", "retorno_condicional"]):
            df = calcular_regime_hmm(df, n_states=n_states)

        df = calcular_microestrutura_proxy(df)
        df = calcular_anomalias(df)
        df = calcular_sharpe_rolling(df)
        df = calcular_autocorrelacao(df)

        # --- Labels por ativo, coerentes com config.label_params ---
        label_cfg = config.get("label_params", {})
        ativo_cfg = label_cfg.get(ativo, {})
        tp_pips = ativo_cfg.get("tp_pips", config.get("tp_pips", 30))
        sl_pips = ativo_cfg.get("sl_pips", config.get("sl_pips", 30))
        janela_label = ativo_cfg.get("janela", config.get("janela_label", 20))

        pip_factors = config.get("pip_factors", {})
        pip_factor = float(pip_factors.get(ativo, 0.0001))

        prior_intrabar = str(config.get("tp_sl_intracandle_priority", "SL")).upper()

        df = gerar_sinal_tp_sl_realista(
            df,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            janela=janela_label,
            pip_factor=pip_factor,
            prioridade_intracandle=prior_intrabar
        )

        df["sinal"] = df["sinal"].apply(normalizar_sinal)
        df = pre_processar_final(df)
        df = _assegurar_timestamp(df)      # garante 'timestamp', 'timestamp_utc' e 'timestamp_epoch'
        auditar_features_ultima_linha(df, ativo=ativo)

        distrib = df["sinal"].value_counts(normalize=True).to_dict()
        log_event(f"[features] Distribuição final dos labels: {distrib}")
        log_event(f"[features] FIM cálculo para {ativo}: {df.shape[0]} linhas finais.")
        return df
    except Exception as e:
        log_event(f"❌ Erro ao calcular features: {e}", level="error")
        try:
            enviar_telegram(f"❌ ERRO calcular_features: {e}")
        except Exception:
            pass
        return pd.DataFrame()


#############################
# Main
#############################

def main():
    log_event(f"[INÍCIO] Gerando features para todos os ativos do config.json...")
    config = carregar_config()
    ativos = config.get("ativos", ["EURUSD"])
    timeframes = config.get("timeframes", {})
    num_candles = config.get("janela_candles", 1000)
    log_event(f"Iniciando geração de features para ativos: {ativos}, {num_candles} candles cada.")

    # Lazy import do MT5 para não quebrar em ambientes sem a lib
    try:
        import MetaTrader5 as mt5  # type: ignore
        TIMEFRAME_MAP = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        HAS_MT5 = True
    except Exception as e:
        HAS_MT5 = False
        TIMEFRAME_MAP = {}
        log_event(f"[features] MetaTrader5 indisponível ({e}). O main() não irá coletar candles.", level="warning")

    try:
        from mt5.coletar_candles_mt5 import coletar_candles  # type: ignore
    except Exception as e:
        coletar_candles = None
        log_event(f"[features] coletar_candles_mt5 indisponível ({e}).", level="warning")

    if not HAS_MT5 or coletar_candles is None:
        log_event("[features] Saindo do main(): dependências MT5 ausentes.", level="warning")
        return

    all_features = []
    for ativo in ativos:
        timeframe_str = timeframes.get(ativo, config.get("timeframe", "M1"))
        timeframe = TIMEFRAME_MAP.get(timeframe_str, None)
        if timeframe is None:
            log_event(f"[features] Timeframe '{timeframe_str}' inválido para {ativo}. Pulando.", level="warning")
            continue

        log_event(f"Coletando candles para {ativo} (timeframe={timeframe_str}, {num_candles} candles)")
        try:
            candles = coletar_candles(ativo, quantidade=num_candles, timeframe=timeframe)
        except Exception as e:
            msg = f"Erro ao coletar candles para {ativo}: {e}"
            log_event(msg, level="error")
            try:
                enviar_telegram(f"❌ {msg}")
            except Exception:
                pass
            continue

        if candles is None or candles.empty:
            msg = f"Não foi possível coletar candles para {ativo} ou arquivo vazio."
            log_event(msg, level="warning")
            continue

        df_feat = calcular_features(candles, config, ativo)
        if df_feat.empty:
            msg = f"Nenhuma feature gerada para {ativo}."
            log_event(msg, level="warning")
            continue

        df_feat["ativo"] = ativo
        all_features.append(df_feat)
        log_event(f"Features calculadas para {ativo} com {df_feat.shape[0]} linhas.")

    if all_features:
        df_all = pd.concat(all_features, ignore_index=True)
        os.makedirs("dados", exist_ok=True)
        # Mantemos apenas numéricos/bool (como antes), mas agora com timestamp_epoch incluso
        df_all["sinal"] = df_all["sinal"].apply(normalizar_sinal)
        numericas = df_all.select_dtypes(include=[np.number, 'bool']).columns
        df_all[numericas].to_csv("dados/features.csv", index=False)
        msg = f"Features salvas em dados/features.csv ({df_all.shape[0]} linhas totais)"
        log_event(msg)
        try:
            enviar_telegram(f"[features] {msg}")
        except Exception:
            pass
    else:
        msg = "Nenhuma feature gerada para os ativos informados. Verifique candles e conexão MT5."
        log_event(msg, level="warning")
        try:
            enviar_telegram(f"[features] {msg}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
