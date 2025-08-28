# inteligencia/estrategia_ia.py
# (Adições não destrutivas: sanity de padrão, gates de contexto e de ensemble)
import pandas as pd
import numpy as np
import os
import joblib

from utils.utils import salvar_log, carregar_config
from utils.debug_logger import log_event
from utils.sinal_utils import normalizar_sinal, sinal_to_str
from utils.thresholds import carregar_thresholds, arquivo_stale
from inteligencia.ranking_padroes import obter_score_padrao
from inteligencia.modo_sniper import detectar_sniper  # ⬅️ GATE SNIPER (configurável)
from utils.vote_monitor import registrar_voto

# Suporte para deep learning e XGBoost
try:
    from tensorflow import keras
except Exception:
    keras = None

try:
    import xgboost as xgb
    from xgboost import DMatrix
except Exception:
    xgb = None

MODELOS_CARREGADOS = {}

# =========================================================
# NOVO: meta-features (iguais às do treino) e listas .pkl
# =========================================================
def adicionar_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ok_close = "close" in df.columns
    ok_h = "high" in df.columns
    ok_l = "low" in df.columns

    if ok_close:
        returns = df["close"].pct_change()
        df["volatility_20"] = returns.rolling(20).std()
        df["volatility_50"] = returns.rolling(50).std()
        df["trend_strength"] = df["close"].rolling(20).apply(
            lambda x: abs(np.polyfit(range(len(x)), x, 1)[0]) / np.std(x) if np.std(x) > 0 else 0,
            raw=False
        )
    if ok_close and ok_h and ok_l:
        roll_mean = df["close"].rolling(20).mean()
        roll_mean = roll_mean.replace(0, np.nan)
        df["range_ratio"] = (df["high"] - df["low"]) / roll_mean
        df["range_ratio"] = df["range_ratio"].fillna(0.0)

    return df


def _load_feature_list(model_key: str):
    pkl_map = {
        "random_forest": "modelos/features_treinadas_rf.pkl",
        "xgboost":       "modelos/features_treinadas_xgb.pkl",
        "lstm":          "modelos/features_treinadas_lstm.pkl",
    }
    pkl_path = pkl_map.get(model_key)
    if pkl_path and os.path.exists(pkl_path):
        try:
            feats = joblib.load(pkl_path)
            feats = list(feats)
            if feats:
                return feats
        except Exception as e:
            log_event(f"[FEATURES] Falha ao carregar {pkl_path}: {e}", level="warning")

    if model_key == "xgboost":
        feats_path = "modelos/xgb_features.json"
        if os.path.exists(feats_path):
            try:
                import json
                with open(feats_path, "r", encoding="utf-8") as f:
                    feats = json.load(f).get("features", [])
                if feats:
                    return feats
            except Exception as e:
                log_event(f"[FEATURES] Falha ao ler {feats_path}: {e}", level="warning")

    return None


# ----------------------------
# Config helpers
# ----------------------------
def _obter_cfg_modelos(config):
    modelos_cfg = config.get("modelos_sinal", None)
    if modelos_cfg is None:
        modelos_cfg = config.get("ensemble_modelos", {})
        if modelos_cfg:
            log_event("[GERAR_SINAL] Usando chave LEGADA 'ensemble_modelos'. Recomenda-se migrar para 'modelos_sinal' no config.json.", level="warning")
    else:
        if "ensemble_modelos" in config:
            log_event("[GERAR_SINAL] 'modelos_sinal' encontrado. Ignorando 'ensemble_modelos'.", level="info")
    if not isinstance(modelos_cfg, dict):
        log_event("[GERAR_SINAL] Config de modelos inválida. Esperado dict.", level="error")
        modelos_cfg = {}
    return modelos_cfg


def _read_params(config):
    return {
        "ensemble_voting": config.get("ensemble_voting", "majority"),  # majority|media|max
        "neutro_weight": float(config.get("neutro_weight", 0.50)),
        "xgb_neutro_cap": float(config.get("xgb_neutro_cap", 0.50)),
        "anti_neutro": bool(config.get("anti_neutro", True)),
        "anti_neutro_threshold": float(config.get("anti_neutro_threshold", 0.55)),
        "anti_neutro_margin": float(config.get("anti_neutro_margin", 0.05)),
        "ensemble_2x1_conf": bool(config.get("ensemble_2x1_conf", True)),
        "ensemble_min_conf": float(config.get("ensemble_min_conf", 0.70)),
        "turbo_override_enable": bool(config.get("turbo_override_enable", True)),
        "turbo_override_conf": float(config.get("turbo_override_conf", 0.80)),
        "turbo_requires_no_squeeze": bool(config.get("turbo_requires_no_squeeze", True)),
        "sniper_scoring_enable": bool(config.get("sniper_scoring_enable", False)),
        "tp_pips": float(config.get("tp_pips", 40)),
        "sl_pips": float(config.get("sl_pips", 20)),
        "ev_rule_enable": bool(config.get("ev_rule_enable", False)),
        "ev_min_pips": float(config.get("ev_min_pips", 1.0)),
    }


# ----------------------------
# Carregamento de modelos
# ----------------------------
def carregar_modelos(config):
    modelos_cfg = _obter_cfg_modelos(config)
    caminhos_modelos_padrao = {
        "random_forest": "modelos/cerebro_mestre.joblib",
        "lstm": "modelos/lstm_cerebro.h5",
        "xgboost": "modelos/xgb_cerebro.json"
    }
    for nome, info in modelos_cfg.items():
        try:
            if isinstance(info, dict) and not info.get("ativo", True):
                continue
        except Exception:
            pass

        caminho = caminhos_modelos_padrao.get(nome)
        if not caminho or not os.path.exists(caminho):
            log_event(f"[GERAR_SINAL] Modelo {nome} ({caminho}) não encontrado.", level="warning")
            continue

        if nome in MODELOS_CARREGADOS:
            continue

        try:
            if caminho.endswith(".joblib"):
                MODELOS_CARREGADOS[nome] = joblib.load(caminho)
            elif caminho.endswith(".h5") and keras:
                MODELOS_CARREGADOS[nome] = keras.models.load_model(caminho)
            elif caminho.endswith(".json") and xgb:
                modelo = xgb.Booster()
                modelo.load_model(caminho)
                MODELOS_CARREGADOS[nome] = modelo
            else:
                log_event(f"[GERAR_SINAL] Extensão desconhecida p/ {caminho}", level="warning")
        except Exception as e:
            log_event(f"[GERAR_SINAL] Falha ao carregar {nome}: {e}", level="error")


# ----------------------------
# Pré-processamento (alinha features às do treino)
# ----------------------------
def preprocessar_features(df, nome_modelo):
    try:
        cfg = carregar_config()
        usar_meta = bool(cfg.get("usar_meta_features", True))
        df_work = df.copy()

        if usar_meta:
            try:
                df_work = adicionar_meta_features(df_work)
            except Exception as e:
                log_event(f"[PREP] adicionar_meta_features falhou: {e}", level="warning")

        feats = _load_feature_list(nome_modelo)

        df_num = df_work.select_dtypes(include=[np.number, "bool"])
        if feats:
            df_align = df_num.reindex(columns=feats)
        else:
            df_align = df_num

        df_align = (
            df_align.tail(1).copy()
                   .ffill()
                   .bfill()
                   .fillna(0.0)
                   .replace([np.inf, -np.inf], 0.0)
                   .astype(np.float32)
        )
        X = df_align.values
        return {"X": X, "features_treinadas": feats, "df_alinhado": df_align}
    except Exception as e:
        log_event(f"[PREP] Erro no preprocessamento: {e}", level="error")
        return None


# ----------------------------
# Preditores
# ----------------------------
def predict_model(modelo, tipo, X):
    if tipo == "random_forest":
        try:
            pred = modelo.predict(X)[0]
            conf = None
            try:
                conf = float(np.max(modelo.predict_proba(X)[0]))
            except Exception:
                conf = 0.5
            return int(normalizar_sinal(pred)), conf, "RandomForest"
        except Exception as e:
            log_event(f"[RF] Erro no predict: {e}", level="error")
            return 0, 0.5, "RandomForest"

    elif tipo == "lstm" and keras:
        try:
            X_use = X
            if X_use.ndim == 2:
                X_use = X_use.reshape((X_use.shape[0], 1, X_use.shape[1]))
            X_use = np.nan_to_num(X_use, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            pred = modelo.predict(X_use, verbose=0)[0]
            if hasattr(pred, "__len__") and len(pred) > 1:
                pred_label = int(np.argmax(pred))
                conf = float(np.max(pred))
            else:
                pred_label = int(np.round(float(pred)))
                conf = float(abs(float(pred)))
            return int(normalizar_sinal(pred_label)), conf, "LSTM"
        except Exception as e:
            log_event(f"[LSTM] Erro no predict: {e}", level="error")
            return 0, 0.5, "LSTM"

    elif tipo == "xgboost" and xgb:
        try:
            dm = DMatrix(X)
            proba = MODELOS_CARREGADOS["xgboost"].predict(dm)[0]
            if isinstance(proba, (list, np.ndarray)) and len(proba) > 1:
                pred_label = int(np.argmax(proba))
                conf = float(np.max(proba))
            else:
                conf = float(abs(float(proba)))
                pred_label = int(np.sign(float(proba)))
            return int(normalizar_sinal(pred_label)), conf, "XGBoost"
        except Exception as e:
            log_event(f"[XGB] Erro no predict: {e}", level="error")
            return 0, 0.5, "XGBoost"

    else:
        return 0, 0.5, tipo


# ----------------------------
# Agregação + Anti-neutro
# ----------------------------
def _agregar_votos(votos, params):
    if not votos:
        return 0, 0.0, 0.0, (-1, 0.0, None), 0.0

    conf_media = float(np.mean([v["conf"] for v in votos]))

    soma = 0.0
    peso = 0.0
    conf_neutro_max_eff = 0.0
    melhor_non_neutro = (-1, 0.0, None)

    for v in votos:
        s = int(normalizar_sinal(v["sinal"]))
        c_eff = float(v.get("conf_eff", v.get("conf", 0.0)))
        c_raw = float(v.get("conf", 0.0))

        if s == 0:
            conf_neutro_max_eff = max(conf_neutro_max_eff, c_eff)
            c_eff *= (1.0 - float(params.get("neutro_weight", 0.50)))
        soma += (s * c_eff)
        peso += abs(c_eff)

        if s != 0 and c_raw > melhor_non_neutro[1]:
            melhor_non_neutro = (s, c_raw, v.get("modelo"))

    score_media_pesada = (soma / peso) if peso > 0 else 0.0
    if params.get("ensemble_voting", "majority") == "max":
        nao_neutros = [v for v in votos if int(normalizar_sinal(v.get("sinal", 0))) != 0]
        if nao_neutros:
            vbest = max(nao_neutros, key=lambda v: float(v.get("conf", 0.0)))
            return int(normalizar_sinal(vbest.get("sinal", 0))), float(vbest.get("conf", 0.0)), conf_media, melhor_non_neutro, conf_neutro_max_eff
        vbest = max(votos, key=lambda v: float(v.get("conf", 0.0)))
        return 0, float(vbest.get("conf", 0.0)), conf_media, melhor_non_neutro, conf_neutro_max_eff

    if score_media_pesada > 0:
        sinal_ensemble = 1
    elif score_media_pesada < 0:
        sinal_ensemble = -1
    else:
        sinal_ensemble = 0

    return sinal_ensemble, abs(float(score_media_pesada)), conf_media, melhor_non_neutro, conf_neutro_max_eff


# ----------------------------
# Fallback técnico
# ----------------------------
def fallback_tecnico(df, contexto_decisao):
    df = df.copy()
    if "close" not in df.columns:
        log_event("[FALLBACK] coluna 'close' ausente; retornando neutro.", level="warning")
        sinal = 0
        saida = {
            "timestamp": df["timestamp"].iloc[-1] if "timestamp" in df.columns else "",
            "sinal": sinal,
            "padrao": "Fallback Técnico",
            "motivo": "Sem coluna close",
            "confianca": 0.5,
            "regime": contexto_decisao.get("regime", "neutro"),
            "contexto": contexto_decisao,
            "tp_sl_priority": "SL",
        }
        sanity_check_sinal(saida)
        return saida

    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["rsi"] = compute_rsi(df["close"], window=14)
    df["bb_upper"], df["bb_lower"] = compute_bbands(df["close"], window=20)
    df["tick_volume"] = df["volume"] if "volume" in df.columns else None

    def decisao(row):
        if row["rsi"] < 30 and row["close"] < row["bb_lower"]:
            return "compra"
        if row["rsi"] > 70 and row["close"] > row["bb_upper"]:
            return "venda"
        if row["close"] > row["sma_20"] and row["sma_20"] > row["sma_50"]:
            return "compra"
        if row["close"] < row["sma_20"] and row["sma_20"] < row["sma_50"]:
            return "venda"
        return "neutro"

    df["sinal"] = df.apply(decisao, axis=1).apply(normalizar_sinal)
    ultimo_sinal = df.loc[df.index[-1], "sinal"]
    padrao_fallback = "Fallback Técnico"
    contexto_decisao["padrao"] = padrao_fallback
    contexto_decisao["confianca"] = 0.5
    contexto_decisao["motivo"] = f"Sinal técnico padrão ({sinal_to_str(ultimo_sinal)})"
    salvar_log(df, "dados/sinais_gerados_fallback.csv")
    log_event(f"Sinal FALLBACK: {sinal_to_str(ultimo_sinal)} | {contexto_decisao}", level="warning")

    saida = {
        "timestamp": df["timestamp"].iloc[-1] if "timestamp" in df.columns else "",
        "sinal": ultimo_sinal,
        "padrao": contexto_decisao["padrao"],
        "motivo": contexto_decisao["motivo"],
        "confianca": contexto_decisao["confianca"],
        "regime": contexto_decisao["regime"],
        "contexto": contexto_decisao,
        "tp_sl_priority": "SL",
    }
    sanity_check_sinal(saida)
    return saida


# ----------------------------
# Pipeline principal
# ----------------------------
def gerar_sinal(df_candles, ativo, contexto=None):
    df = df_candles.copy()
    df["sinal"] = 0
    config = carregar_config()
    carregar_modelos(config)

    modelos_cfg = _obter_cfg_modelos(config)
    P = _read_params(config)

    # Optionally sobrepor com thresholds do walk-forward
    if bool(config.get("usar_thresholds_walkforward", True)):
        try:
            tau_star, delta_star = carregar_thresholds(path="logs/walkforward_summary.json")
            P["anti_neutro_threshold"] = float(tau_star)
            P["anti_neutro_margin"] = float(delta_star)
            if arquivo_stale(path="logs/walkforward_summary.json", dias=int(config.get("stale_thresholds_dias", 7))):
                log_event("[THRESHOLDS] walkforward_summary.json está desatualizado (stale).", level="warning")
        except Exception as _e:
            log_event(f"[THRESHOLDS] Falha ao carregar thresholds: {_e}", level="warning")

    # ---- NOVO: filtros/gates do config (defaults seguros) ----
    filtros = (config.get("filtros", {}) if isinstance(config, dict) else {}) or {}
    banir_squeeze = bool(filtros.get("banir_squeeze", True))
    banir_vol_baixa = bool(filtros.get("banir_vol_baixa", True))
    consenso_min = int(filtros.get("consenso_min", 2))
    prob_min_abs = float(filtros.get("prob_min_absoluto", 0.62))
    margem_neutro = float(filtros.get("margem_neutro", 0.05))

    # helper de retorno neutro padronizado
    def _saida_neutra(motivo_extra: str):
        contexto_decisao["motivo"] = (contexto_decisao.get("motivo", "") + f" | {motivo_extra}").strip(" |")
        saida_local = {
            "timestamp": df["timestamp"].iloc[-1] if "timestamp" in df.columns else "",
            "sinal": 0,
            "padrao": contexto_decisao.get("padrao"),
            "motivo": contexto_decisao["motivo"],
            "confianca": contexto_decisao.get("confianca", 0.5),
            "regime": contexto_decisao.get("regime", "neutro"),
            "contexto": contexto_decisao,
            "tp_sl_priority": "SL",
        }
        sanity_check_sinal(saida_local)
        return saida_local

    contexto_decisao = {
        "padrao": None,
        "confianca": 0.5,
        "regime": contexto.get("regime", "neutro") if contexto else "neutro",
        "squeeze": contexto.get("squeeze", False) if contexto else False,
        "volatilidade": (contexto.get("volatilidade") if isinstance(contexto, dict) else None),
        "motivo": "",
        "padrao_score": 0.0,
    }

    votos = []
    motivos = []

    # Ranking de padrão (interpretação)
    try:
        contexto_decisao["padrao"], contexto_decisao["padrao_score"] = obter_score_padrao(df.tail(50))
    except Exception as e:
        log_event(f"[PADRAO] não foi possível calcular: {e}", level="warning")

    # ---- NOVO: SANITY de padrão (bloqueante) ----
    try:
        ps = contexto_decisao.get("padrao_score", None)
        if (contexto_decisao.get("padrao") is None) or (ps is None) or (isinstance(ps, float) and np.isnan(ps)):
            log_event("[GATE] NEUTRO — padrao_invalido", level="info")
            return _saida_neutra("padrao_invalido")
    except Exception as _e:
        log_event(f"[GATE] erro ao validar padrao: {_e}", level="warning")
        return _saida_neutra("padrao_invalido")

    # ---- NOVO: GATES de contexto (squeeze/volatilidade) ----
    try:
        if banir_squeeze and bool(contexto_decisao.get("squeeze", False)):
            log_event("[GATE] NEUTRO — squeeze_ativo", level="info")
            return _saida_neutra("squeeze")
        vol = str(contexto_decisao.get("volatilidade", "")).lower()
        if banir_vol_baixa and vol == "baixa":
            log_event("[GATE] NEUTRO — volatilidade_baixa", level="info")
            return _saida_neutra("volatilidade_baixa")
    except Exception as _e:
        log_event(f"[GATE] erro nos gates de contexto: {_e}", level="warning")

    # Para cada modelo configurado e carregado
    for nome, info in modelos_cfg.items():
        modelo = MODELOS_CARREGADOS.get(nome)
        if not modelo:
            motivos.append(f"{nome}:NAO_CARREGADO")
            continue

        prep = preprocessar_features(df, nome)
        if prep is None:
            motivos.append(f"{nome}:ERRO_FEATURES")
            continue

        X = prep["X"]
        feats = prep["features_treinadas"]
        df_feat_alinhado = prep["df_alinhado"]

        try:
            if nome == "xgboost" and xgb:
                pred, conf = _predict_xgb_com_nomes(df_feat_alinhado, feats, "modelos/xgb_cerebro.json")
                modelo_nome = "XGBoost"
            elif nome == "random_forest":
                pred_label = int(modelo.predict(df_feat_alinhado)[0])
                try:
                    conf = float(np.max(modelo.predict_proba(df_feat_alinhado)[0]))
                except Exception:
                    conf = 0.5
                pred = int(normalizar_sinal(pred_label))
                modelo_nome = "RandomForest"
            else:
                pred, conf, modelo_nome = predict_model(modelo, nome, X)

            s = int(normalizar_sinal(pred))
            conf_raw = float(conf)

            conf_eff = conf_raw
            if s == 0:
                if "xgb" in nome.lower() or "xgboost" in modelo_nome.lower():
                    conf_eff = min(conf_eff, float(P.get("xgb_neutro_cap", 0.50)))
                conf_eff *= (1.0 - float(P.get("neutro_weight", 0.50)))

            votos.append({
                "modelo": modelo_nome,
                "nome_key": nome,
                "sinal": s,
                "conf": conf_raw,
                "conf_eff": float(conf_eff),
            })
            log_event(f"[VOTO] {ativo} {modelo_nome} sinal={s} conf={conf_raw:.3f} conf_eff={conf_eff:.3f}", level="info")

        except Exception as e:
            motivos.append(f"{nome}:ERRO({e})")
            log_event(f"[GERAR_SINAL] Erro ao prever com {nome}: {e}", level="error")

    if not votos:
        log_event(f"Nenhuma previsão válida dos modelos (motivos: {';'.join(motivos) if motivos else '—'}). Usando fallback técnico.", level="warning")
        return fallback_tecnico(df, contexto_decisao)

    # ## SNIPER scoring (não-invasivo)
    if P.get('sniper_scoring_enable', False):
        try:
            sc, sc_max = _sniper_score(contexto_decisao if isinstance(contexto_decisao, dict) else {}, P.get('sniper_score_weights', {}))
            contexto_decisao['sniper_score'] = sc
            contexto_decisao['sniper_score_max'] = sc_max
            log_event(f"[SNIPER] score={sc}/{sc_max}", level='info')
        except Exception as _e:
            log_event(f"[SNIPER] scoring falhou: {_e}", level='warning')

    # Agregação
    sinal_final, score_media_pesada, conf_media, melhor_non_neutro, conf_neutro_max_eff = _agregar_votos(votos, P)

    # ---- NOVO: GATES do ensemble (antes de promoções) ----
    if sinal_final != 0:
        try:
            total = len(votos)
            votos_lado = sum(1 for v in votos if int(v["sinal"]) == int(sinal_final))
            votos_pos = sum(1 for v in votos if int(v["sinal"]) == 1)
            votos_neg = sum(1 for v in votos if int(v["sinal"]) == -1)
            frac = abs((votos_pos - votos_neg)) / max(1, total)

            if votos_lado < consenso_min:
                log_event(f"[GATE] NEUTRO — consenso({votos_lado})<{consenso_min}", level="info")
                return _saida_neutra("consenso_insuficiente")
            if conf_media < prob_min_abs:
                log_event(f"[GATE] NEUTRO — conf_media {conf_media:.3f}<{prob_min_abs:.2f}", level="info")
                return _saida_neutra("prob_baixa")
            if frac < margem_neutro:
                log_event(f"[GATE] NEUTRO — margem {frac:.3f}<{margem_neutro:.2f}", level="info")
                return _saida_neutra("margem_neutro")
        except Exception as _e:
            log_event(f"[GATE] erro ao checar gates do ensemble: {_e}", level="warning")

    # Turbo override (se existir um voto com conf muito alta e sem squeeze, prioriza)
    if bool(P.get("turbo_override_enable", True)) and contexto_decisao.get("squeeze", False) is False:
        if melhor_non_neutro[1] >= float(P.get("turbo_override_conf", 0.80)):
            log_event(f"[TURBO] override por conf alta do melhor não-neutro ({melhor_non_neutro[1]:.3f}).", level="info")
            sinal_final = int(melhor_non_neutro[0])

    # Regras 2x1 por confiança
    if bool(P.get("ensemble_2x1_conf", True)) and sinal_final == 0:
        lado_pos = sum(1 for v in votos if int(v['sinal']) == 1 and float(v['conf']) >= float(P.get("ensemble_min_conf", 0.70)))
        lado_neg = sum(1 for v in votos if int(v['sinal']) == -1 and float(v['conf']) >= float(P.get("ensemble_min_conf", 0.70)))
        if lado_pos >= 2 and lado_pos > lado_neg:
            sinal_final = 1
            contexto_decisao["motivo"] = (contexto_decisao.get("motivo","") + " | 2x1_conf").strip(" |")
        elif lado_neg >= 2 and lado_neg > lado_pos:
            sinal_final = -1
            contexto_decisao["motivo"] = (contexto_decisao.get("motivo","") + " | 2x1_conf").strip(" |")

    # --- Anti-neutro (pós-ensemble) ---
    if P["anti_neutro"] and sinal_final == 0 and melhor_non_neutro[1] >= P["anti_neutro_threshold"]:
        if (melhor_non_neutro[1] - conf_neutro_max_eff) >= P["anti_neutro_margin"]:
            log_event(f"[ANTI-NEUTRO] promovido modelo={melhor_non_neutro[2]} conf={melhor_non_neutro[1]:.3f} > neutro_max_eff={conf_neutro_max_eff:.3f}", level="info")
            sinal_final = int(melhor_non_neutro[0])

    # --- GATE SNIPER (veto a entradas sem confluência) ---
    usar_sniper = bool(config.get("usar_modo_sniper", True))
    sniper_ok = True
    if usar_sniper and sinal_final != 0:
        regime_for_sniper = (contexto.get("regime") if isinstance(contexto, dict) else None) or "any"
        try:
            sniper_ok = detectar_sniper(df, ativo, contexto=contexto or {}, regime=regime_for_sniper)
        except Exception as e:
            log_event(f"[SNIPER] Erro no gate: {e}", level="error")
            sniper_ok = True
        if not sniper_ok:
            log_event(f"[SNIPER] bloqueou entrada em {ativo} (sem confluência).", level="info")
            return _saida_neutra("sniper_block")

    # ## EV gate (final, opcional)
    if bool(P.get('ev_rule_enable', False)) and sinal_final != 0:
        try:
            probas = contexto_decisao.get('probas', {}) if isinstance(contexto_decisao, dict) else {}
            p_plus = float(probas.get('+1', probas.get(1, float('nan'))))
            p_minus = float(probas.get('-1', probas.get(-1, float('nan'))))
            def _ev_pips_local(p_plus, p_minus, tp, sl):
                ev_long = p_plus*tp - p_minus*sl
                ev_short = p_minus*tp - p_plus*sl
                return ev_long, ev_short
            if p_plus == p_plus and p_minus == p_minus:
                ev_long, ev_short = _ev_pips_local(p_plus, p_minus, float(P.get('tp_pips', 40)), float(P.get('sl_pips', 20)))
                ev_min = float(P.get('ev_min_pips', 1.0))
                ev_side = ev_long if sinal_final == 1 else ev_short
                if ev_side < ev_min:
                    log_event(f"[EV] bloqueado | EV={ev_side:.2f} < {ev_min:.2f} (p+={p_plus:.2f}, p-={p_minus:.2f})", level='info')
                    return _saida_neutra("ev_block")
        except Exception as _e:
            log_event(f"[EV] erro no gate: {_e}", level='warning')

    padrao_nome = "|".join([f"{v['modelo']}:{v['sinal']}" for v in votos])
    try:
        contexto_decisao["confianca"] = float(conf_media)
    except Exception:
        contexto_decisao["confianca"] = 0.5
    contexto_decisao["motivo"] = (contexto_decisao.get("motivo", "") + f" | votos={padrao_nome}").strip(" |")

    saida = {
        "timestamp": df["timestamp"].iloc[-1] if "timestamp" in df.columns else "",
        "sinal": int(normalizar_sinal(sinal_final)),
        "padrao": contexto_decisao["padrao"],
        "motivo": contexto_decisao["motivo"],
        "confianca": contexto_decisao["confianca"],
        "regime": contexto_decisao["regime"],
        "contexto": contexto_decisao,
        "tp_sl_priority": "SL",
    }
    sanity_check_sinal(saida)
    try:
        registrar_voto(ativo, int(normalizar_sinal(sinal_final)), config)
    except Exception as _e:
        log_event(f"[VOTE_MONITOR] falha ao registrar voto: {_e}", level="warning")
    return saida


# ----------------------------
# Utilidades locais
# ----------------------------
def sanity_check_sinal(d: dict):
    problemas = []
    for k in ("sinal", "confianca", "motivo", "padrao"):
        if k not in d or d[k] is None or (isinstance(d[k], str) and d[k] == ""):
            problemas.append(k)
    if problemas:
        log_event(f"[SANITY CHECK SINAL] Campos essenciais nulos/vazios: {problemas} | Dados: {d}", level="warning")


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_bbands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (num_std * std)
    lower = sma - (num_std * std)
    return upper, lower


def _sniper_score(contexto, pesos):
    try:
        t = bool(contexto.get('tendencia_ok', False))
        v = bool(contexto.get('vol_ok', False)) or (str(contexto.get('volatilidade','')).lower() in ('media','alta'))
        ns = not bool(contexto.get('squeeze', False))
        sp = bool(contexto.get('spread_ok', True))
        score = (pesos.get('tendencia',1.0)*(1 if t else 0) +
                 pesos.get('vol_ok',1.0)*(1 if v else 0) +
                 pesos.get('no_squeeze',1.0)*(1 if ns else 0) +
                 pesos.get('spread_ok',1.0)*(1 if sp else 0))
        max_score = float(pesos.get('tendencia',1.0) + pesos.get('vol_ok',1.0) + pesos.get('no_squeeze',1.0) + pesos.get('spread_ok',1.0))
        return float(score), float(max_score)
    except Exception:
        return 0.0, 4.0


def _predict_xgb_com_nomes(df_feat_alinhado, feats, caminho_modelo):
    if xgb is None:
        raise RuntimeError("XGBoost não disponível")
    if feats is None:
        X = df_feat_alinhado.values
        feat_names = list(df_feat_alinhado.columns)
    else:
        X = df_feat_alinhado[feats].values
        feat_names = feats
    dm = DMatrix(X, feature_names=feat_names)
    proba = MODELOS_CARREGADOS["xgboost"].predict(dm)[0]
    if isinstance(proba, (list, np.ndarray)) and len(proba) > 1:
        pred_label = int(np.argmax(proba))
        conf = float(np.max(proba))
    else:
        conf = float(abs(float(proba)))
        pred_label = int(np.sign(float(proba)))
    return int(normalizar_sinal(pred_label)), conf
