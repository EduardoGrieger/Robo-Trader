import pandas as pd
import joblib
import numpy as np
import os

try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    from tensorflow import keras
except ImportError:
    keras = None

features_path = "dados/features.csv"
modelo_rf_path = "modelos/cerebro_mestre.joblib"
modelo_xgb_path = "modelos/xgb_cerebro.json"
modelo_lstm_path = "modelos/lstm_cerebro.h5"

def carregar_dados(caminho):
    try:
        df = pd.read_csv(caminho)
        return df
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

def mapear_label_xgb(y):
    return np.where(y == -1, 0, np.where(y == 0, 1, 2))

def desmapear_label_xgb(y_xgb):
    return np.where(y_xgb == 0, -1, np.where(y_xgb == 1, 0, 1))

def get_features_usados(path_pkl, df):
    """Carrega e alinha as features corretas"""
    if os.path.exists(path_pkl):
        features = joblib.load(path_pkl)
        # Garante que todos os features existam no df
        features = [f for f in features if f in df.columns]
    else:
        features = list(df.select_dtypes(include=[np.number, 'bool']).columns)
    return features

def prever_sinais_rf(modelo_path, df):
    if not os.path.exists(modelo_path):
        print("Modelo RandomForest não encontrado.")
        return None
    try:
        modelo = joblib.load(modelo_path)
        features = get_features_usados("modelos/features_treinadas_rf.pkl", df)
        X = df[features].fillna(0)
        sinais = modelo.predict(X)
        print(f"RF - Features usados: {len(features)} - {features}")
        return pd.Series(sinais, index=df.index)
    except Exception as e:
        print(f"Erro ao prever RF: {e}")
        return None

def prever_sinais_xgb(modelo_path, df):
    if not xgb or not os.path.exists(modelo_path):
        print("XGBoost não instalado ou modelo não encontrado.")
        return None
    try:
        modelo = xgb.Booster()
        modelo.load_model(modelo_path)
        features = get_features_usados("modelos/features_treinadas_xgb.pkl", df)
        X = df[features].fillna(0)
        dmatrix = xgb.DMatrix(X)
        sinais_prob = modelo.predict(dmatrix)
        if sinais_prob.ndim > 1 and sinais_prob.shape[1] > 1:
            sinais = np.argmax(sinais_prob, axis=1)
        else:
            sinais = np.round(sinais_prob).astype(int)
        sinais = desmapear_label_xgb(sinais)
        print(f"XGB - Features usados: {len(features)} - {features}")
        return pd.Series(sinais, index=df.index)
    except Exception as e:
        print(f"Erro ao prever XGB: {e}")
        return None

def prever_sinais_lstm(modelo_path, df):
    if not keras or not os.path.exists(modelo_path):
        print("Keras não instalado ou modelo não encontrado.")
        return None
    try:
        modelo = keras.models.load_model(modelo_path)
        features = get_features_usados("modelos/features_treinadas_lstm.pkl", df)
        X = df[features].fillna(0).astype(np.float32).values
        X_lstm = np.expand_dims(X, axis=1)
        sinais_prob = modelo.predict(X_lstm, verbose=0)
        if sinais_prob.ndim > 1 and sinais_prob.shape[1] > 1:
            sinais = np.argmax(sinais_prob, axis=1)
            sinais = np.where(sinais == 0, -1, np.where(sinais == 1, 0, 1))
        else:
            sinais = np.round(sinais_prob).astype(int)
        print(f"LSTM - Features usados: {len(features)} - {features}")
        return pd.Series(sinais, index=df.index)
    except Exception as e:
        print(f"Erro ao prever LSTM: {e}")
        return None

def sinais_fallback(df):
    if all(col in df.columns for col in ["close", "sma_20"]):
        sinais = df.apply(
            lambda x: 1 if x["close"] > x["sma_20"]
            else (-1 if x["close"] < x["sma_20"] else 0),
            axis=1
        )
        print("Fallback - usando close vs sma_20.")
        return sinais
    else:
        print("Colunas necessárias para fallback não encontradas.")
        return None

def main():
    df = carregar_dados(features_path)
    if df is None:
        return

    print("\nResumo de previsões dos modelos:")

    sinais_rf = prever_sinais_rf(modelo_rf_path, df)
    if sinais_rf is not None:
        print(f"RandomForest: {sinais_rf.value_counts().to_dict()}")

    sinais_xgb = prever_sinais_xgb(modelo_xgb_path, df)
    if sinais_xgb is not None:
        print(f"XGBoost: {sinais_xgb.value_counts().to_dict()}")

    sinais_lstm = prever_sinais_lstm(modelo_lstm_path, df)
    if sinais_lstm is not None:
        print(f"LSTM: {sinais_lstm.value_counts().to_dict()}")

    sinais_fb = sinais_fallback(df)
    if sinais_fb is not None:
        print(f"Fallback SMA20: {sinais_fb.value_counts().to_dict()}")

    print("\nResumo: Quantidade de sinais de compra/venda/neutro em cada abordagem para os últimos dados.")

if __name__ == "__main__":
    main()
