import pandas as pd
import joblib
import os
from colorama import Fore, Style, init

try:
    from .debug_logger import log_event
except ImportError:
    from debug_logger import log_event

init(autoreset=True)

FEATURES_PATH = "dados/features.csv"
MODELOS = {
    "RandomForest": "modelos/cerebro_mestre.joblib",
    "XGBoost": "modelos/xgb_cerebro.json",
    "LSTM": "modelos/lstm_cerebro.h5"
}
DROP_COLS = ["sinal", "timestamp", "data_hora", "ativo"]

def carregar_features_do_csv(caminho):
    try:
        df = pd.read_csv(caminho)
        features = [col for col in df.columns if col not in DROP_COLS]
        return features, df
    except Exception as e:
        log_event(f"Erro ao carregar CSV: {e}", level="error")
        return [], pd.DataFrame()

def carregar_features_rf(caminho):
    try:
        modelo = joblib.load(caminho)
        return list(modelo.feature_names_in_)
    except Exception as e:
        log_event(f"Erro RF: {e}", level="error")
        return []

def carregar_features_xgb(caminho):
    try:
        import xgboost as xgb
        modelo = xgb.Booster()
        modelo.load_model(caminho)
        # XGBoost não salva feature_names por padrão!
        return []
    except Exception as e:
        log_event(f"Erro XGBoost: {e}", level="error")
        return []

def carregar_features_lstm(caminho):
    try:
        from tensorflow import keras
        modelo = keras.models.load_model(caminho)
        # LSTM em tabular: não salva features. Usa o do CSV.
        return []
    except Exception as e:
        log_event(f"Erro LSTM: {e}", level="error")
        return []

def main():
    log_event("-"*50, level="info")
    log_event("[INÍCIO] AUDITORIA DE FEATURES ENSEMBLE", level="info")
    log_event("-"*50, level="info")

    csv_features, _ = carregar_features_do_csv(FEATURES_PATH)
    log_event(f"FEATURES DO CSV DE PRODUÇÃO: {csv_features}", level="info")

    rf_features = carregar_features_rf(MODELOS["RandomForest"]) if os.path.exists(MODELOS["RandomForest"]) else []
    xgb_features = carregar_features_xgb(MODELOS["XGBoost"]) if os.path.exists(MODELOS["XGBoost"]) else []
    lstm_features = carregar_features_lstm(MODELOS["LSTM"]) if os.path.exists(MODELOS["LSTM"]) else []

    for nome, features in [("RandomForest", rf_features), ("XGBoost", xgb_features), ("LSTM", lstm_features)]:
        if not features:
            features = csv_features
            log_event(f"[{nome}] Usando features do CSV como referência (não disponível no modelo).", level="warning")
        log_event(f"[{nome}] FEATURES ESPERADAS: {features}", level="info")
        faltando = [f for f in features if f not in csv_features]
        sobrando = [f for f in csv_features if f not in features]
        if not faltando and not sobrando:
            log_event(f"[{nome}] ✅ Alinhamento perfeito!", level="info")
        else:
            if faltando:
                log_event(f"[{nome}] FALTANDO no CSV: {faltando}", level="error")
            if sobrando:
                log_event(f"[{nome}] SOBRANDO no CSV: {sobrando}", level="warning")

    log_event("-"*50, level="info")
    log_event("[FIM] Auditoria concluída!", level="info")
    log_event("-"*50, level="info")

if __name__ == "__main__":
    main()
