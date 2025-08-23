# inteligencia/estrategia_ia.py
import pandas as pd
import numpy as np
import os
import joblib

from utils.utils import salvar_log, carregar_config
from utils.debug_logger import log_event
from utils.sinal_utils import normalizar_sinal, sinal_to_str
from utils.thresholds import carregar_thresholds, arquivo_stale
from inteligencia.ranking_padroes import obter_score_padrao
from inteligencia.modo_sniper import detectar_sniper  # ⬅️ NOVO: gate sniper

# Suporte para deep learning e XGBoost
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    keras = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None

MODELOS_CARREGADOS = {}

# ----------------------------
# Config helpers
# ----------------------------
def _obter_cfg_modelos(config):
    """
    Compatibilidade: aceita tanto 'modelos_sinal' (preferido) quanto 'ensemble_modelos' (legado).
    """
    modelos_cfg = config.get("modelos_sinal", None)
    if modelos_cfg is None:
        modelos_cfg = config.get("ensemble_modelos", {})
        if modelos_cfg:
            log_event("[GERAR_SINAL] Usando chave LEGADA 'ensemble_modelos'. "
                      "Recomenda-se migrar para 'modelos_sinal' no config.json.", level="warning")
    else:
        if "ensemble_modelos" in config:
            log_event("[GERAR_SINAL] 'modelos_sinal' encontrado. Ignorando 'ensemble_modelos'.", level="info")
    if not isinstance(modelos_cfg, dict):
        log_event("[GERAR_SINAL] Config de modelos inválida. Esperado dict.", level="error")
        modelos_cfg = {}
    return modelos_cfg

def _read_params(config):
    """Lê parâmetros usados na decisão/anti-neutro com defaults seguros."""
    return {
        "ensemble_voting": config.get("ensemble_voting", "majority"),
        "threshold_confianca_turbo": float(config.get("threshold_confianca_turbo", 0.30)),
        "neutro_weight": float(config.get("neutro_weight", 0.60)),
        "xgb_neutro_cap": float(config.get("xgb_neutro_cap", 0.70)),
        "anti_neutro": bool(config.get("anti_neutro", True)),
        # Ajuste padrão mais permissivo para sair do 0
        "anti_neutro_threshold": float(config.get("anti_neutro_threshold", 0.65)),
        "anti_neutro_margin": float(config.get("anti_neutro_margin", 0.10)),
        # Mantido por compatibilidade, mas NÃO mais usado aqui para zerar sinal
        "limiar_score_padrao": float(config.get("limiar_score_padrao", 0.00)),
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
                log_event(f"[GERAR_SINAL] Extensão não suportada ou lib ausente: {caminho}", level="warning")
        except Exception as e:
            log_event(f"[GERAR_SINAL] Erro ao carregar modelo {nome}: {e}", level="error")

# ----------------------------
# Preprocessamento de features
# ----------------------------
def preprocessar_features(df, modelo_nome):
    """
    Carrega a lista de features utilizada no treino do modelo específico e
    prepara a última amostra para predição.
    - Protegido contra colunas faltantes: reindex e fillna(0).
    - Para LSTM, retorna shape (batch, timesteps, features) = (1, 1, N).
    - Retorna também o DataFrame alinhado com nomes, para uso do XGBoost.
    """
    caminho_features = f"modelos/features_treinadas_{modelo_nome}.pkl"
    if not os.path.exists(caminho_features):
        log_event(f"[GERAR_SINAL] Features treinadas não encontradas para {modelo_nome} ({caminho_features})", level="error")
        return None

    try:
        features_treinadas = joblib.load(caminho_features)
    except Exception as e:
        log_event(f"[GERAR_SINAL] Falha ao ler features_treinadas_{modelo_nome}.pkl: {e}", level="error")
        return None

    try:
        df_ult = df.tail(1).copy()
        df_filtrado = df_ult.reindex(columns=features_treinadas, fill_value=0).fillna(0)
        X = df_filtrado.values
        if modelo_nome == "lstm":
            X = np.expand_dims(X, axis=1)  # (1,1,N)
        return {
            "X": X,
            "features_treinadas": features_treinadas,
            "df_alinhado": df_filtrado
        }
    except Exception as e:
        log_event(f"[GERAR_SINAL] Erro ao montar features para {modelo_nome}: {e}", level="error")
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
            pred = modelo.predict(X, verbose=0)[0]
            if pred.ndim > 0 and len(pred) > 1:
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
            dmatrix = xgb.DMatrix(X)
            pred = modelo.predict(dmatrix)
            if isinstance(pred, np.ndarray) and pred.ndim > 1 and pred.shape[1] > 1:
                pred_label = int(np.argmax(pred, axis=1)[0])
                conf = float(np.max(pred))
            else:
                val = float(pred[0])
                pred_label = int(np.round(val))
                conf = float(abs(val))
            label_map = {0: -1, 1: 0, 2: 1}
            pred_final = label_map.get(pred_label, 0)
            return int(normalizar_sinal(pred_final)), conf, "XGBoost"
        except Exception as e:
            log_event(f"[XGB] Erro no predict: {e}", level="error")
            return 0, 0.5, "XGBoost"

    return 0, 0.5, "Desconhecido"

def _predict_xgb_com_nomes(df_alinhado, lista_features, caminho_modelo="modelos/xgb_cerebro.json"):
    """
    Faz predict no Booster do XGB garantindo feature_names no DMatrix.
    Mapeamento de classes usado no treino:
      -1 -> 0,  0 -> 1,  1 -> 2
    Inverso na inferência:
       0 -> -1, 1 -> 0,  2 -> 1
    """
    try:
        import xgboost as xgb  # import local
        X_df = df_alinhado.reindex(columns=lista_features).astype(float).tail(1)
        dmat = xgb.DMatrix(X_df, feature_names=lista_features)
        booster = xgb.Booster()
        booster.load_model(caminho_modelo)

        prob = booster.predict(dmat)  # (1, num_class) em multi:softprob
        if hasattr(prob, "shape") and len(prob.shape) == 2 and prob.shape[1] >= 3:
            idx = int(np.argmax(prob[0]))
            mapa_inv = {0: -1, 1: 0, 2: 1}
            classe = mapa_inv.get(idx, 0)
            conf = float(prob[0, idx])
            return classe, conf

        idx = int(prob[0]) if isinstance(prob, np.ndarray) else int(prob)
        mapa_inv = {0: -1, 1: 0, 2: 1}
        return mapa_inv.get(idx, 0), 0.5

    except Exception as e:
        log_event(f"[XGB] Erro no predict: {e}", level="error")
        return 0, 0.5

# ----------------------------
# Agregação + Anti-neutro
# ----------------------------
def _agregar_votos(votos, params):
    """
    votos: lista de dicts:
      { "modelo": "XGBoost"/"RandomForest"/"LSTM",
        "nome_key": "xgboost"/"random_forest"/"lstm",
        "sinal": int(-1/0/1), "conf": float, "conf_eff": float }
    Retorna: sinal_ensemble (int), score_media_pesada (float), conf_media (float),
             melhor_non_neutro: (sinal, conf, modelo), conf_neutro_max_eff (float)
    """
    if not votos:
        return 0, 0.0, 0.0, (-1, 0.0, None), 0.0

    # média de confiança "bruta"
    conf_media = float(np.mean([v["conf"] for v in votos]))

    # média pesada por confiança (reduzindo peso do 0)
    soma = 0.0
    peso = 0.0
    conf_neutro_max_eff = 0.0
    melhor_non_neutro = (-1, 0.0, None)

    for v in votos:
        s = int(normalizar_sinal(v["sinal"]))
        c_eff = float(v["conf_eff"])
        w = c_eff
        if s == 0:
            conf_neutro_max_eff = max(conf_neutro_max_eff, c_eff)
        else:
            if v["conf"] > melhor_non_neutro[1]:
                melhor_non_neutro = (s, float(v["conf"]), v["modelo"])
        soma += s * w
        peso += w

    score_media_pesada = (soma / peso) if peso > 0 else 0.0
    # threshold suave para sair do 0
    if score_media_pesada > 0.15:
        sinal_mp = 1
    elif score_media_pesada < -0.15:
        sinal_mp = -1
    else:
        sinal_mp = 0

    return sinal_mp, score_media_pesada, conf_media, melhor_non_neutro, conf_neutro_max_eff

# ----------------------------
# Fallback técnico
# ----------------------------
def fallback_tecnico(df, contexto_decisao):
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
        "tp_sl_priority": "SL",  # default
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
    P.setdefault('ensemble_2x1_conf', True)
    P.setdefault('ensemble_min_conf', 0.70)
    P.setdefault('turbo_override_enable', True)
    P.setdefault('turbo_override_conf', 0.80)
    P.setdefault('turbo_requires_no_squeeze', True)
    P.setdefault('turbo_volume_multiplier', 0.50)
    P.setdefault('sniper_scoring_enable', False)
    P.setdefault('sniper_score_weights', {'tendencia':1.0,'vol_ok':1.0,'no_squeeze':1.0,'spread_ok':1.0})
    P.setdefault('sniper_score_trade_min', 2)
    P.setdefault('sniper_score_microlote_if_one', True)
    P.setdefault('ensemble_min_conf', 0.70)
    P.setdefault('turbo_override_conf', 0.80)

    # Carrega thresholds aprendidos (tau/delta) do walk-forward, se existir
    if bool(config.get("usar_thresholds_walkforward", True)):
        try:
            tau_star, delta_star = carregar_thresholds(path="logs/walkforward_summary.json")
            # Ajusta limiares do anti-neutro a partir do walk-forward
            P["anti_neutro_threshold"] = float(tau_star)
            P["anti_neutro_margin"] = float(delta_star)
            if arquivo_stale(path="logs/walkforward_summary.json", dias=int(config.get("stale_thresholds_dias", 7))):
                log_event("[THRESHOLDS] walkforward_summary.json está desatualizado (stale).", level="warning")
        except Exception as _e:
            log_event(f"[THRESHOLDS] Falha ao carregar thresholds: {_e}", level="warning")
    contexto_decisao = {
        "padrao": None,
        "confianca": 0.5,
        "regime": contexto.get("regime", "neutro") if contexto else "neutro",
        "motivo": ""
    }

    votos = []   # por-modelo
    motivos = [] # strings curtas por modelo

    # --- Previsão por modelo ---
    for nome, info in modelos_cfg.items():
        try:
            if isinstance(info, dict) and not info.get("ativo", True):
                continue
        except Exception:
            pass

        modelo = MODELOS_CARREGADOS.get(nome)
        if not modelo:
            motivos.append(f"{nome}:NÃO_CARREGADO")
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

            # Aplica caps/pesos para efeito no ensemble (não altera conf_raw)
            conf_eff = conf_raw
            if s == 0:
                if "xgb" in nome.lower() or "xgb" in modelo_nome.lower():
                    conf_eff = min(conf_eff, P["xgb_neutro_cap"])
                conf_eff = conf_eff * P["neutro_weight"]

            votos.append({
                "modelo": modelo_nome,
                "nome_key": nome,
                "sinal": s,
                "conf": conf_raw,
                "conf_eff": conf_eff,
            })
            motivos.append(f"{nome}:{s}|Conf:{conf_raw:.2f}|ConfEff:{conf_eff:.2f}")

            log_event(f"[VOTO] modelo={modelo_nome} sinal={s} conf={conf_raw:.3f} conf_eff={conf_eff:.3f}", level="info")

        except Exception as e:
            motivos.append(f"{nome}:ERRO({e})")
            log_event(f"[GERAR_SINAL] Erro ao prever com {nome}: {e}", level="error")

    # Se não houve votos válidos → fallback
    if not votos:
        log_event(f"Nenhuma previsão válida dos modelos (motivos: {';'.join(motivos) if motivos else '—'}). "
                  f"Usando fallback técnico.", level="warning")
    # ## SNIPER scoring (não-invasivo)
    if P.get('sniper_scoring_enable', False):
        try:
            sc, sc_max = _sniper_score(contexto_decisao if isinstance(contexto_decisao, dict) else {}, P.get('sniper_score_weights', {}))
            contexto_decisao['sniper_score'] = sc
            contexto_decisao['sniper_score_max'] = sc_max
            log_event(f"[SNIPER] score={sc}/{sc_max}", level='info')
        except Exception as _e:
            log_event(f"[SNIPER] scoring falhou: {_e}", level='warning')

    # ## EV gate (final, opcional) — aplica no resultado antes do retorno
    if P.get('ev_rule_enable', False) and sinal_final != 0:
        try:
            probas = contexto_decisao.get('probas', {}) if isinstance(contexto_decisao, dict) else {}
            p_plus = float(probas.get('+1', probas.get(1, float('nan'))))
            p_minus = float(probas.get('-1', probas.get(-1, float('nan'))))
            def _ev_pips_local(p_plus, p_minus, tp, sl):
                ev_long = p_plus*tp - p_minus*sl
                ev_short = p_minus*tp - p_plus*sl
                return ev_long, ev_short
            if p_plus==p_plus and p_minus==p_minus:
                ev_long, ev_short = _ev_pips_local(p_plus, p_minus, P.get('tp_pips', 40), P.get('sl_pips', 20))
                ev_min = float(P.get('ev_min_pips', 1.0))
                ev_side = ev_long if sinal_final==1 else ev_short
                if ev_side < ev_min:
                    log_event(f"[EV] bloqueado | EV={ev_side:.2f} < min={ev_min:.2f} (p+={p_plus:.2f}, p-={p_minus:.2f})", level='info')
                    sinal_final = 0
                    contexto_decisao['motivo'] = (contexto_decisao.get('motivo','') + ' | ev_block').strip(' |')
            else:
                log_event("[EV] probabilidades indisponíveis; regra ignorada", level='warning')
        except Exception as _e:
            log_event(f"[EV] falhou: {_e}", level='warning')


        return fallback_tecnico(df, contexto_decisao)

    # --- Agregação (respeita ensemble_voting) + média pesada para referência ---
    sinal_media_pesada, score_mp, conf_media, melhor_non_neutro, conf_neutro_max_eff = _agregar_votos(votos, P)

    preds = [v["sinal"] for v in votos]
    confs = [v["conf"] for v in votos]

    if P["ensemble_voting"] == "majority":
        values, counts = np.unique(preds, return_counts=True)
        sinal_final = int(values[np.argmax(counts)])
    elif P["ensemble_voting"] == "media":
        sinal_final = int(np.sign(score_mp))
    elif P["ensemble_voting"] == "max":
        sinal_final = preds[int(np.argmax(confs))]
    else:
        sinal_final = int(np.sign(sum(preds)))

    confianca_final = float(np.mean(confs))
    # --- Regra 2x1 com confiança mínima ---
    if P.get("ensemble_2x1_conf", True) and sinal_final == 0:
        # Agrupa votos por sinal != 0 com conf mínima
        votos_validos = {}
        for v in votos:
            s = int(v.get("sinal", 0))
            c = float(v.get("conf", 0.0))
            if s != 0 and c >= float(P.get("ensemble_min_conf", 0.70)):
                votos_validos.setdefault(s, []).append((v["modelo"], c))
        # Se houver algum lado com >=2 modelos, escolhe o de maior soma de confs
        candidatos = [(s, sum(c for _,c in lst), lst) for s,lst in votos_validos.items() if len(lst) >= 2]
        if candidatos:
            candidatos.sort(key=lambda x: x[1], reverse=True)
            s_escolhido, soma_conf, lst = candidatos[0]
            log_event(f"[ENSEMBLE 2x1] Forçando sinal={s_escolhido} por maioria 2x1 com conf mínima "
                      f"(detalhe: {[(m, round(c,3)) for m,c in lst]})", level="info")
            sinal_final = int(s_escolhido)
            contexto_decisao["motivo"] = (contexto_decisao.get("motivo","") + " | 2x1_conf").strip(" |")


    # --- Anti-neutro (pós-ensemble) ---
    if P["anti_neutro"] and sinal_final == 0 and melhor_non_neutro[1] >= P["anti_neutro_threshold"]:
        if (melhor_non_neutro[1] - conf_neutro_max_eff) >= P["anti_neutro_margin"]:
            log_event(f"[ANTI-NEUTRO] promovido modelo={melhor_non_neutro[2]} "
                      f"conf={melhor_non_neutro[1]:.3f} > neutro_max_eff={conf_neutro_max_eff:.3f}", level="info")
            sinal_final = int(melhor_non_neutro[0])

    # --- GATE SNIPER (veto a entradas sem confluência) ---
    usar_sniper = bool(config.get("usar_modo_sniper", True))
    sniper_ok = True
    if usar_sniper and sinal_final != 0:
        # se não vier um 'regime' explícito no contexto, passamos 'any' para não bloquear à toa
        regime_for_sniper = (contexto.get("regime") if isinstance(contexto, dict) else None) or "any"
        try:
            sniper_ok = detectar_sniper(df, ativo, contexto=contexto or {}, regime=regime_for_sniper)
        except Exception as e:
            log_event(f"[SNIPER] Erro no gate: {e}", level="error")
            sniper_ok = True  # não bloqueia em caso de erro
        if not sniper_ok:
            motivos.append("SNIPER:BLOCK")
            log_event(f"[SNIPER] bloqueou entrada em {ativo} (sem confluência).", level="info")
            sinal_final = 0
        else:
            motivos.append("SNIPER:OK")

    # --- Monta motivo/padrão e (apenas) anota score de reforço ---
    padrao_nome = "|".join([f"{v['modelo']}:{v['sinal']}" for v in votos])
    contexto_decisao["padrao"] = padrao_nome
    contexto_decisao["confianca"] = round(confianca_final, 2)
    contexto_decisao["motivo"] = "|".join(motivos) + f"|mp={score_mp:.3f}"

    score_reforco = obter_score_padrao(padrao_nome, ativo)
    if score_reforco is not None:
        contexto_decisao["motivo"] += f"; Score_Reforco={score_reforco}"

    # Log final e saída
    df.loc[df.index[-1], "sinal"] = sinal_final
    salvar_log(df, "dados/sinais_gerados_ensemble.csv")
    log_event(f"Sinal ENSEMBLE: {sinal_to_str(sinal_final)} | {contexto_decisao}", level="info")

    saida = {
        "timestamp": df["timestamp"].iloc[-1] if "timestamp" in df.columns else "",
        "sinal": sinal_final,
        "padrao": contexto_decisao["padrao"],
        "motivo": contexto_decisao["motivo"],
        "confianca": contexto_decisao["confianca"],
        "regime": contexto_decisao["regime"],
        "contexto": contexto_decisao,
        "tp_sl_priority": str(config.get("tp_sl_intracandle_priority", "SL")).upper(),  # ⬅️ NOVO
    }
    sanity_check_sinal(saida)
    return saida

# ----------------------------
# Sanity / Indicadores simples
# ----------------------------
def sanity_check_sinal(d):
    campos = ["timestamp", "sinal", "padrao", "motivo", "confianca", "regime"]
    problemas = []
    for k in campos:
        v = d.get(k)
        if v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and str(v).strip() == ""):
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
        sp = bool(contexto.get('spread_ok', True))  # assume ok se não tiver
        score = (pesos.get('tendencia',1.0)*(1 if t else 0) +
                 pesos.get('vol_ok',1.0)*(1 if v else 0) +
                 pesos.get('no_squeeze',1.0)*(1 if ns else 0) +
                 pesos.get('spread_ok',1.0)*(1 if sp else 0))
        max_score = sum([pesos.get('tendencia',1.0), pesos.get('vol_ok',1.0), pesos.get('no_squeeze',1.0), pesos.get('spread_ok',1.0)])
        return float(score), float(max_score)
    except Exception:
        return 0.0, 4.0
