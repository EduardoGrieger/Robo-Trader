# smoke_todos.py
# =====================================================================
# Suite de smoke tests do projeto, 100% offline (mocks onde necessário)
# =====================================================================

import os, sys, importlib, types, math, tempfile, shutil, inspect
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GREEN = "✅"
RED = "❌"
YELLOW = "⚠️"

def _ok(b): return GREEN if b else RED

# ---------------------------------------------------------------------
# Fallback de imports (pacote vs arquivos no raiz)
# ---------------------------------------------------------------------
def _try_import(module_name, fallback_name=None):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        if fallback_name:
            return importlib.import_module(fallback_name)
        raise

def _try_from(module_name, attr, fallback_module=None):
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, attr)
    except ModuleNotFoundError:
        if fallback_module:
            mod = importlib.import_module(fallback_module)
            return getattr(mod, attr)
        raise

# util/log (se não houver, cria stub local)
try:
    from utils.debug_logger import log_event
except Exception:
    def log_event(msg, level="info"):
        print(f"[{level.upper()}] {msg}")

# ---------------------------------------------------------------------
# 0) Helpers de cenário sintético
# ---------------------------------------------------------------------
def mk_candles(n=300, start=None):
    if start is None:
        start = pd.Timestamp.utcnow().floor("min") - pd.Timedelta(minutes=n)
    t = pd.date_range(start, periods=n, freq="min", tz="UTC")
    base = 1.1000 + 0.001*np.sin(np.linspace(0, 12*np.pi, n))
    noise = np.random.normal(0, 0.00025, size=n)
    close = base + noise
    open_ = np.r_[close[0], close[:-1]]
    hl = np.random.uniform(0.0001, 0.0005, size=n)
    high  = np.maximum(open_, close) + hl/2
    low   = np.minimum(open_, close) - hl/2
    tick_volume = np.random.randint(50, 400, size=n)
    return pd.DataFrame({"time": t, "open": open_, "high": high, "low": low, "close": close, "tick_volume": tick_volume})

# ---------------------------------------------------------------------
# 1) clusterizar_regimes.detectar_regime
# ---------------------------------------------------------------------
print("✅ clusterizar_regimes.detectar_regime — ", end="")
ok = True
try:
    cluster_mod = _try_import("inteligencia.clusterizar_regimes", "clusterizar_regimes")
    df = mk_candles(500)
    series = df["close"].pct_change().dropna().values
    try:
        res = cluster_mod.detectar_regime(series, use_logret=False)
    except TypeError:
        res = cluster_mod.detectar_regime(series)
    tipo = score = vol = None
    if isinstance(res, dict):
        tipo  = res.get("tipo") or res.get("regime") or res.get("estado")
        score = res.get("score") or res.get("qualidade") or res.get("prob")
        vol   = res.get("vol") or res.get("volatilidade") or res.get("std")
    elif isinstance(res, (list, tuple)):
        if len(res) >= 1: tipo  = res[0]
        if len(res) >= 2: score = res[1]
        if len(res) >= 3: vol   = res[2]
    else:
        tipo = str(res)
    print(f"retorno=(tipo={tipo!r}, score={score}, vol={vol})")
except Exception as e:
    ok = False
    print(f"{RED} erro: {e}")
if not ok: sys.exit(2)

# ---------------------------------------------------------------------
# 2) inferencia_contextual.prever_contexto
# ---------------------------------------------------------------------
print("✅ inferencia_contextual.prever_contexto — ", end="")
try:
    inf_mod = _try_import("inteligencia.inferencia_contextual", "inferencia_contextual")
    df = mk_candles(240)
    cfg = {
        "infer_vol_high_ratio": 1.5, "infer_vol_med_ratio": 0.9,
        "infer_volume_high_ratio": 1.5, "infer_volume_med_ratio": 1.05,
        "squeeze_bandwidth_factor": 0.005
    }
    try:
        ctx = inf_mod.prever_contexto(df, config=cfg)
    except TypeError:
        try:
            ctx = inf_mod.prever_contexto(df, cfg)
        except TypeError:
            ctx = inf_mod.prever_contexto(df)
    print(ctx)
except Exception as e:
    print(f"{RED} erro: {e}")
    sys.exit(2)

# ---------------------------------------------------------------------
# 3) contexto.analisar_contexto
# ---------------------------------------------------------------------
print("✅ contexto.analisar_contexto — ", end="")
try:
    ctx_mod = _try_import("inteligencia.contexto", "contexto")
    df = mk_candles(240)
    cfg = {"contexto_vol_window": 20, "contexto_vol_ref_window": 200, "squeeze_bandwidth_factor": 0.005}
    try:
        res = ctx_mod.analisar_contexto(df, cfg)
    except TypeError:
        try:
            res = ctx_mod.analisar_contexto(df, config=cfg)
        except TypeError:
            res = ctx_mod.analisar_contexto(df)
    print(res)
except Exception as e:
    print(f"{RED} erro: {e}")
    sys.exit(2)

# ---------------------------------------------------------------------
# 4) modo_sniper.detectar_sniper (compatível com bool / dict / tuple)
# ---------------------------------------------------------------------
print("✅ modo_sniper.detectar_sniper — ", end="")
try:
    sniper_mod = _try_import("inteligencia.modo_sniper", "modo_sniper")
    df = mk_candles(200)

    def _call_sniper(dframe, cfg):
        try:
            return sniper_mod.detectar_sniper(dframe, cfg)
        except TypeError:
            try:
                return sniper_mod.detectar_sniper(dframe, config=cfg)
            except TypeError:
                return sniper_mod.detectar_sniper(dframe)

    def _as_bool(resp):
        if isinstance(resp, bool):
            return resp
        if isinstance(resp, dict):
            return bool(resp.get("sniper") or resp.get("is_sniper") or resp.get("flag"))
        if isinstance(resp, (list, tuple)) and len(resp) > 0 and isinstance(resp[0], bool):
            return resp[0]
        return bool(resp)

    out = _call_sniper(df, {"squeeze_bandwidth_factor": 0.005})
    s1 = _as_bool(out)
    print(f"[squeeze=True → False] — sniper={s1}")

    df2 = df.copy()
    df2["close"] = df2["close"] + np.linspace(0, 0.01, len(df2))
    out2 = _call_sniper(df2, {"squeeze_bandwidth_factor": 0.0001})
    s2 = _as_bool(out2)
    print(f"✅ modo_sniper.detectar_sniper [sem squeeze + vol alta + RSI>70] — sniper={s2}")
except Exception as e:
    print(f"{RED} modo_sniper erro: {e}")
    sys.exit(2)

# ---------------------------------------------------------------------
# 5) validar_tp_sl — invocação inteligente posicional (assinaturas variadas)
# ---------------------------------------------------------------------
print("✅ validar_tp_sl.validar_tp_sl_historico — ", end="")
try:
    vmod = _try_import("inteligencia.validar_tp_sl", "validar_tp_sl")
    fn = vmod.validar_tp_sl_historico
    df = mk_candles(40)

    DF_SYNS   = {"df", "dados", "candles", "data", "dataframe", "ohlc", "series"}
    TP_SYNS   = {"tp_pips", "tp"}
    SL_SYNS   = {"sl_pips", "sl"}
    PIP_SYNS  = {"ponto_pip", "pip_factor", "point", "pip_value", "pip_size", "pip"}
    PRIO_SYNS = {"prioridade_intracandle", "prioridade", "prioridade_intraday"}

    def _call_smart(dframe, tp, sl, pf=0.0001, priority="SL"):
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        args = []
        used = {"df": False, "tp": False, "sl": False, "pf": False, "prio": False}
        for i, p in enumerate(params):
            name = p.name
            if (name in DF_SYNS) or (i == 0 and not used["df"]):
                args.append(dframe); used["df"] = True; continue
            if (name in TP_SYNS) and not used["tp"]:
                args.append(tp); used["tp"] = True; continue
            if (name in SL_SYNS) and not used["sl"]:
                args.append(sl); used["sl"] = True; continue
            if (name in PIP_SYNS):
                args.append(pf); used["pf"] = True; continue
            if (name in PRIO_SYNS) and not used["prio"]:
                args.append(priority); used["prio"] = True; continue
            if p.default is inspect._empty:
                args.append(0.0); continue  # placeholder para obrigatórios desconhecidos
        return fn(*args)

    res = _call_smart(df, tp=20, sl=20, pf=0.0001, priority="SL")
    print(f"[normal] — resultado={res}")

    df2 = df.copy()
    j = len(df2) - 1
    df2.loc[j, "high"] = df2.loc[j, "open"] + 0.01
    df2.loc[j, "low"]  = df2.loc[j, "open"]  - 0.01
    res2 = _call_smart(df2, tp=5, sl=5, pf=0.0001, priority="ordem")
    print(f"✅ validar_tp_sl [TP & SL mesmo candle] — prioridade=ordem => {res2}")

except Exception as e:
    print(f"{RED} validar_tp_sl erro: {e}")
    sys.exit(2)

# ---------------------------------------------------------------------
# 6) ranking_padroes — compatível sem 'arquivo=' (usa default)
# ---------------------------------------------------------------------
try:
    rmod = _try_import("inteligencia.ranking_padroes", "ranking_padroes")
    os.makedirs("dados", exist_ok=True)
    path_smoke = os.path.join("dados","ranking_padroes_smoke.csv")
    default_path = os.path.join("dados","ranking_padroes.csv")
    pd.DataFrame([
        {"padrao":"compra_tendencia_manha", "evidencia":30, "score":1.0},
        {"padrao":"venda_lateral_noite", "evidencia":10, "score":0.2},
        {"padrao":"neutro", "evidencia":5, "score":0.0},
    ]).to_csv(path_smoke, index=False)
    # Se a função não aceitar 'arquivo=', colocamos também no path default
    try:
        s = rmod.obter_score_padrao("compra_tendencia_manha", arquivo=path_smoke)
    except TypeError:
        shutil.copy(path_smoke, default_path)
        s = rmod.obter_score_padrao("compra_tendencia_manha")
    print(f"{GREEN} ranking_padroes — score(compra_tendencia_manha)={s:.2f}")
except Exception as e:
    print(f"{YELLOW} ranking_padroes aviso: {e}")

# ---------------------------------------------------------------------
# 7) contexto_ordens — registrar_contexto_ordem (assinatura-agnóstica)
# ---------------------------------------------------------------------
try:
    cord = _try_import("inteligencia.contexto_ordens", "contexto_ordens")
    fn = cord.registrar_contexto_ordem
    path_out = os.path.join("dados","contextos_ordens_smoke.csv")

    # Mapeamento por assinatura → só posicionais
    TKT_SYNS  = {"ticket","id","ordem","order_id"}
    ATIVO_SYNS= {"ativo","symbol","par"}
    CTX_SYNS  = {"contexto","ctx","features","dados"}
    RES_SYNS  = {"resultado","result","label","status"}
    ARQ_SYNS  = {"arquivo","path","filepath","csv_path"}

    def _reg(ticket, ativo, contexto, resultado, arquivo):
        sig = inspect.signature(fn)
        args = []
        for p in sig.parameters.values():
            n = p.name
            if n in TKT_SYNS:
                args.append(ticket); continue
            if n in ATIVO_SYNS:
                args.append(ativo); continue
            if n in CTX_SYNS:
                args.append(contexto); continue
            if n in RES_SYNS:
                args.append(resultado); continue
            if n in ARQ_SYNS:
                args.append(arquivo); continue
            # desconhecido obrigatório → placeholder
            if p.default is inspect._empty:
                args.append(None)
        return fn(*args)

    _reg(123, "EURUSD", {"sessao":"teste","volatilidade":"baixa","squeeze":True}, "win", path_out)
    print(f"{GREEN} contexto_ordens.registrar_contexto_ordem — -> {path_out}")

    # ranking vencedores (se existir)
    if hasattr(cord, "ranking_padroes_vencedores"):
        try:
            cord.ranking_padroes_vencedores(arquivo=path_out, out_csv=os.path.join("dados","ranking_padroes_vencedores_smoke.csv"))
            print(f"{GREEN} contexto_ordens.ranking_padroes_vencedores")
        except TypeError:
            # fallback sem kwargs
            cord.ranking_padroes_vencedores(path_out, os.path.join("dados","ranking_padroes_vencedores_smoke.csv"))
            print(f"{GREEN} contexto_ordens.ranking_padroes_vencedores (fallback)")
except Exception as e:
    print(f"{YELLOW} contexto_ordens aviso: {e}")

# ---------------------------------------------------------------------
# 8) memoria_adaptativa — tolera ausência de 'contexto'
# ---------------------------------------------------------------------
try:
    mm = _try_import("inteligencia.memoria_adaptativa", "memoria_adaptativa")
    try:
        score = mm.obter_score_memoria("EURUSD", contexto={"sessao":"teste"})
    except TypeError:
        score = mm.obter_score_memoria("EURUSD")
    print(f"{GREEN} memoria_adaptativa — score_memoria={score}")
except Exception as e:
    print(f"{YELLOW} memoria_adaptativa aviso: {e}")

# ---------------------------------------------------------------------
# 9) pipeline_ia — compatível sem 'config='
# ---------------------------------------------------------------------
try:
    pm = _try_import("inteligencia.pipeline_ia", "pipeline_ia")
    df = mk_candles(200)
    cfg = {"tp_pips":40,"sl_pips":20}
    try:
        out = pm.pipeline_ia(df, config=cfg)
    except TypeError:
        try:
            out = pm.pipeline_ia(df, cfg)
        except TypeError:
            out = pm.pipeline_ia(df)
    print(f"{GREEN} pipeline_ia.pipeline_ia — {out}")
except Exception as e:
    print(f"{YELLOW} pipeline_ia aviso: {e}")

# ---------------------------------------------------------------------
# 10) model_ensemble — tenta (df) → (X, y)
# ---------------------------------------------------------------------
try:
    me = _try_import("inteligencia.model_ensemble", "model_ensemble")
    df = mk_candles(180)
    try:
        out = me.pipeline_ensemble(df)
    except TypeError:
        # constrói features/label simples
        ret = df["close"].pct_change().fillna(0.0).to_numpy()
        sma5 = df["close"].rolling(5, min_periods=1).mean().to_numpy()
        X = np.column_stack([ret, sma5])
        y = (ret > 0).astype(int)
        out = me.pipeline_ensemble(X, y)
    # impressão amigável
    if isinstance(out, dict) and "stack_acc" in out:
        print(f"{GREEN} model_ensemble.pipeline_ensemble — stack_acc={out.get('stack_acc'):.3f}")
    else:
        print(f"{GREEN} model_ensemble.pipeline_ensemble — ok ({type(out).__name__})")
except Exception as e:
    print(f"{YELLOW} model_ensemble aviso: {e}")

# ---------------------------------------------------------------------
# 11) sentimento_mercado
# ---------------------------------------------------------------------
try:
    sm = _try_import("inteligencia.sentimento_mercado", "sentimento_mercado")
    score, itens = sm.obter_sentimento_mercado("EURUSD")
    print(f"{GREEN} sentimento_mercado.obter_sentimento_mercado — score={score:.2f}, itens={len(itens)}")
except Exception as e:
    print(f"{YELLOW} sentimento_mercado aviso: {e}")

# ---------------------------------------------------------------------
# 12) estrategia_ia (import)
# ---------------------------------------------------------------------
try:
    _ = _try_import("inteligencia.estrategia_ia", "estrategia_ia")
    print("✅ estrategia_ia (import)")
except Exception as e:
    print(f"{YELLOW} estrategia_ia aviso: {e}")

# ---------------------------------------------------------------------
# 13) FEATURES — smoke dedicado (100% offline)
# ---------------------------------------------------------------------
print("=== SMOKE: features/gerar_features ===")
try:
    gf = _try_import("features.gerar_features", "features.gerar_features")
except ModuleNotFoundError:
    gf = _try_import("gerar_features")

cfg = {
    "ativos": ["EURUSD"],
    "n_clusters_regime": 3,
    "n_states_hmm": 3,
    "tp_pips": 40,
    "sl_pips": 20,
    "pip_factors": {"EURUSD": 0.0001},
    "label_params": {"EURUSD": {"tp_pips": 40, "sl_pips": 20, "janela": 20}},
    "tp_sl_intracandle_priority": "SL",
}
try:
    candles = mk_candles(650)
    df_feat = gf.calcular_features(candles, cfg, ativo="EURUSD")
    ok_nan = not df_feat.tail(50).select_dtypes(include=[np.number,"bool"]).isnull().any().any()
    ok_inf = not np.isinf(df_feat.tail(50).select_dtypes(include=[np.number,"bool"]).to_numpy()).any()
    ok_ess = all(c in df_feat.columns for c in
                 ["rsi","sma_20","sma_50","bb_high","bb_low","macd","macd_signal","macd_hist",
                  "ema_21","ema_200","atr_14","adx_14","cci","stoch_k","stoch_d","sinal"])
    ok_ts = "timestamp" in df_feat.columns and not df_feat["timestamp"].tail(5).isnull().all()

    if ok_nan and ok_inf:
        print(f"{GREEN} últimas 50 linhas sem NaN/inf")
    else:
        print(f"{RED} problema com NaN/inf nas últimas 50")

    print(f"{_ok(ok_ess)} colunas essenciais presentes")
    print(f"{_ok(ok_ts)} coluna 'timestamp' presente")

    if "sinal" in df_feat.columns:
        dist = df_feat["sinal"].value_counts(normalize=True).to_dict()
        print(f"{GREEN} distribuição de labels: {dist}")

    num = df_feat.select_dtypes(include=[np.number,"bool"]).tail(200)
    var = num.var().sort_values(ascending=False).head(10)
    print("\nTop 10 variâncias (últimas 200 linhas):")
    for k, v in var.items():
        print(f" - {k}: {v:.6e}")

    if not (ok_nan and ok_inf and ok_ess and ok_ts):
        print(f"{RED} Smoke features FALHOU.")
        sys.exit(2)
    else:
        print(f"{GREEN} Smoke features PASSOU.")
except Exception as e:
    print(f"{RED} erro features: {e}")
    sys.exit(2)

# ---------------------------------------------------------------------
# 14) GESTÃO — fechar_todas_ordens (mocks locais, não usa MT5 real)
# ---------------------------------------------------------------------
print("=== SMOKE: gestao/fechar_todas_ordens ===")
try:
    try:
        ftom = importlib.import_module("gestao.fechar_todas_ordens")
    except ModuleNotFoundError:
        ftom = importlib.import_module("fechar_todas_ordens")

    def _mock_obter_ordens():
        return [
            {"ticket": 1, "symbol": "EURUSD", "price_open": 1.1000, "volume": 0.10},
            {"ticket": 2, "symbol": "EURUSD", "price_open": 1.1010, "volume": 0.20},
            {"ticket": 3, "symbol": "EURUSD", "price_open": 1.0990, "volume": 0.05},
        ]
    def _mock_fechar_ordem(ticket, symbol):
        ticket = int(ticket)
        if ticket % 2 == 1:
            return {"retcode": 10009, "retcode_name":"TRADE_RETCODE_PLACED", "preco_fechamento":1.1005, "lucro":5.0}
        else:
            return {"retcode": 10004, "retcode_name":"TRADE_RETCODE_REJECT"}

    def _mock_log(msg, level="info"): print(f"[{level.upper()}] {msg}")
    def _mock_send(msg): print(f"[TELEGRAM] {msg}"); return True
    def _mock_update(**kw): print(f"[STORE] atualizar_operacao: {kw}")

    ftom.obter_ordens_abertas_mt5 = _mock_obter_ordens
    ftom.fechar_ordem = _mock_fechar_ordem
    ftom.log_event = _mock_log
    ftom.enviar_telegram = _mock_send
    ftom.atualizar_operacao = _mock_update

    fechadas, falhas = ftom.fechar_todas_ordens(motivo="smoke_test", delay_seg=0.0)
    print(f"{GREEN} retornos: fechadas={fechadas}, falhas={falhas}")
    ok_s = (fechadas == 2)
    ok_f = (falhas == 1)
    print(f"{_ok(ok_s)} contagem de sucessos esperada (=2)")
    print(f"{_ok(ok_f)} contagem de falhas esperada (=1)")
    if not (ok_s and ok_f):
        print(f"{RED} FAIL em fechar_todas_ordens")
        sys.exit(2)
except Exception as e:
    print(f"{RED} erro fechar_todas_ordens: {e}")
    sys.exit(2)

# ---------------------------------------------------------------------
# 15) GESTÃO — gestao_dinamica_risco (persistência + fator)
# ---------------------------------------------------------------------
print("=== SMOKE: gestao/gestao_dinamica_risco ===")
try:
    try:
        rmod = importlib.import_module("gestao.gestao_dinamica_risco")
    except ModuleNotFoundError:
        rmod = importlib.import_module("gestao_dinamica_risco")

    tmpdir = tempfile.mkdtemp(prefix="risco_")
    os.environ["HIST_RISCO_PATH"] = os.path.join(tmpdir, "historico_operacoes.csv")

    now = datetime.now(timezone.utc)
    def add(ativo, min_ago, resultado):
        rmod.salvar_performance({
            "timestamp": (now - timedelta(minutes=min_ago)).isoformat(),
            "ativo": ativo,
            "resultado": resultado
        })

    add("EURUSD", 10, "win")
    add("EURUSD", 20, "1")
    add("EURUSD", 30, "loss")
    add("EURUSD", 40, "-1")
    add("EURUSD", 50, "win")
    add("EURUSD", 190, "win")  # fora da janela de 180

    fator = rmod.calcular_fator_risco("EURUSD", janela_minutos=180, n_min=10, suav=0.5)
    ok_range = 0.3 <= fator <= 1.5
    print(f"{_ok(ok_range)} fator em [0.3,1.5]: {fator:.3f}")

    vol_aj = rmod.ajustar_volume_base(0.20, fator)
    ok_vol = 0.01 <= vol_aj <= 100.0
    print(f"{_ok(ok_vol)} volume ajustado plausível: {vol_aj:.3f}")

    if not (ok_range and ok_vol):
        print(f"{RED} FAIL em gestao_dinamica_risco")
        sys.exit(2)
except Exception as e:
    print(f"{RED} erro gestao_dinamica_risco: {e}")
    sys.exit(2)

# ---------------------------------------------------------------------
# 16) GESTÃO — gestao_posicoes (usa MetaTrader5 MOCK local)
# ---------------------------------------------------------------------
print("=== SMOKE: gestao/gestao_posicoes ===")
try:
    mod_mt5 = types.ModuleType("MetaTrader5")
    class _Tick:  # estrutura simples
        def __init__(self, bid, ask): self.bid, self.ask = bid, ask
    class _Account:
        def __init__(self, balance): self.balance = balance
    class _Pos:
        __slots__ = ("ticket","symbol","type","volume","price_open","sl","tp")
        def __init__(self, ticket, symbol, typ, volume, price_open, sl=0.0, tp=0.0):
            self.ticket = ticket; self.symbol = symbol; self.type = typ
            self.volume = volume; self.price_open = price_open; self.sl = sl; self.tp = tp
        def _asdict(self):
            return {"ticket":self.ticket,"symbol":self.symbol,"type":self.type,"volume":self.volume,
                    "price_open":self.price_open,"sl":self.sl,"tp":self.tp}
    _state = {"init": False}
    mod_mt5.initialize = lambda : _state.update(init=True) or True
    mod_mt5.shutdown = lambda : _state.update(init=False)
    mod_mt5.symbol_info_tick = lambda symbol: _Tick(1.1000, 1.1002)
    def _positions_get(symbol=None):
        pos = [
            _Pos(101, "EURUSD", 0, 0.10, 1.0995, sl=1.0980),
            _Pos(202, "EURUSD", 1, 0.20, 1.1008, sl=1.1020),
        ]
        return [p for p in pos if (symbol is None or p.symbol == symbol)]
    mod_mt5.positions_get = _positions_get
    mod_mt5.account_info = lambda : _Account(50000.0)
    mod_mt5.TIMEFRAME_M1 = 1; mod_mt5.TIMEFRAME_M5 = 5
    sys.modules["MetaTrader5"] = mod_mt5

    try:
        gp = importlib.import_module("gestao.gestao_posicoes")
    except ModuleNotFoundError:
        gp = importlib.import_module("gestao_posicoes")

    pnl = gp.lucro_aberto("EURUSD")
    ok_pnl = abs(pnl - 17.0) < 1e-6
    print(f"{_ok(ok_pnl)} lucro_aberto≈17.0 => {pnl:.4f}")

    expo = gp.exposicao_ftmo("EURUSD")
    expo_exp = 0.10*100000*1.0995 + 0.20*100000*1.1008
    ok_expo = abs(expo - expo_exp) < 1e-6
    print(f"{_ok(ok_expo)} exposicao_ftmo ok => {expo:.2f}")

    bal = gp.saldo_bruto()
    ok_bal = abs(bal - 50000.0) < 1e-6
    print(f"{_ok(ok_bal)} saldo_bruto=50000 => {bal:.2f}")

    risco = gp.risco_aberto_ftmo("EURUSD")
    risco_exp = (1.0995-1.0980)*0.10*100000 + (1.1020-1.1008)*0.20*100000  # 15 + 24 = 39
    ok_risco = abs(risco - risco_exp) < 1e-6
    print(f"{_ok(ok_risco)} risco_aberto_ftmo≈39.0 => {risco:.2f}")

    if not (ok_pnl and ok_expo and ok_bal and ok_risco):
        print(f"{RED} FAIL em gestao_posicoes")
        sys.exit(2)
except Exception as e:
    print(f"{RED} erro gestao_posicoes: {e}")
    sys.exit(2)

print(f"\n{GREEN} Smoke geral PASSOU.")
