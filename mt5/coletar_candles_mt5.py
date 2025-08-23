import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from utils.debug_logger import log_event
from utils.utils import carregar_config
from comunicacao.telegram_alertas import enviar_telegram

def coletar_candles(ativo, quantidade=96, timeframe=mt5.TIMEFRAME_M1):
    """
    Coleta candles do MT5. Retorna DataFrame ou vazio se falhar.
    Loga todos os passos e falhas.
    """
    log_event(f"[COLETOR] Iniciando coleta de candles para {ativo} | tf={timeframe} | quantidade={quantidade}", level="info")

    if not mt5.initialize():
        msg = "[COLETOR] ERRO: Não foi possível inicializar o MetaTrader 5."
        log_event(msg, level="error")
        try:
            enviar_telegram(msg)
        except:
            pass
        return pd.DataFrame()

    try:
        agora = datetime.now()
        candles = mt5.copy_rates_from(ativo, timeframe, agora, quantidade)
        if candles is None or len(candles) == 0:
            msg = f"[COLETOR] ERRO: MT5 retornou vazio para {ativo}. Verifique conexão, ativo e timeframe."
            log_event(msg, level="warning")
            try:
                enviar_telegram(msg)
            except:
                pass
            return pd.DataFrame()
        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s")
        log_event(f"[COLETOR] OK: {len(df)} candles recebidos para {ativo} (tf={timeframe})", level="info")

        # Salva CSV institucional (um arquivo para cada ativo)
        os.makedirs("dados", exist_ok=True)
        features_path = os.path.join("dados", f"features_{ativo}.csv")
        df.to_csv(features_path, index=False)
        log_event(f"[COLETOR] Features salvas em: {features_path}", level="info")

        # Sanity check de colunas essenciais
        essenciais = ["timestamp", "open", "high", "low", "close", "tick_volume"]
        faltando = [col for col in essenciais if col not in df.columns]
        if faltando:
            log_event(f"[COLETOR] WARNING: Colunas faltando após coleta: {faltando}", level="warning")
            return pd.DataFrame()
        return df[essenciais]
    except Exception as e:
        msg = f"[COLETOR] ERRO: Exceção ao coletar candles: {e}"
        log_event(msg, level="error")
        try:
            enviar_telegram(msg)
        except:
            pass
        return pd.DataFrame()
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    config = carregar_config()
    ativos = config.get("ativos", ["EURUSD"])
    timeframes_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    timeframes = config.get("timeframes", {})
    janela_candles = config.get("janela_candles", 1000)

    for ativo in ativos:
        timeframe_str = timeframes.get(ativo, "M1")
        timeframe = timeframes_map.get(timeframe_str, mt5.TIMEFRAME_M1)
        coletar_candles(ativo, quantidade=janela_candles, timeframe=timeframe)
