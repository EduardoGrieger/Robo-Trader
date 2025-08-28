import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone
from utils.debug_logger import log_event
from utils.utils import carregar_config
from comunicacao.telegram_alertas import enviar_telegram

def coletar_candles(ativo, quantidade=96, timeframe=mt5.TIMEFRAME_M1):
    """
    Coleta candles do MT5. Retorna DataFrame com colunas essenciais:
    ['timestamp','open','high','low','close','tick_volume'].

    Ajustes desta versão:
    - 'timestamp' é UTC-aware (pd.Timestamp com tz=UTC)
    - Ordenação crescente pelo tempo + reset de índice
    - Saneamento defensivo de NaNs/linhas incompletas
    """
    log_event(f"[COLETOR] Iniciando coleta de candles para {ativo} | tf={timeframe} | quantidade={quantidade}", level="info")

    if not mt5.initialize():
        msg = "[COLETOR] ERRO: Não foi possível inicializar o MetaTrader 5."
        log_event(msg, level="error")
        try:
            enviar_telegram(msg)
        except Exception:
            pass
        return pd.DataFrame()

    try:
        agora = datetime.now()  # ponto de referência; MT5 usa este 'agora' para cutoff
        candles = mt5.copy_rates_from(ativo, timeframe, agora, quantidade)

        if candles is None or len(candles) == 0:
            msg = f"[COLETOR] ERRO: MT5 retornou vazio para {ativo}. Verifique conexão, ativo e timeframe."
            log_event(msg, level="warning")
            try:
                enviar_telegram(msg)
            except Exception:
                pass
            return pd.DataFrame()

        df = pd.DataFrame(candles)

        # === Timestamp em UTC (aware) ===
        # A estrutura do MT5 traz 'time' como epoch (segundos). Convertemos para tz=UTC.
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)

        # Saneamento mínimo: remove linhas quebradas e ordena pelo tempo
        base_cols = ["timestamp", "open", "high", "low", "close", "tick_volume"]
        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        log_event(f"[COLETOR] OK: {len(df)} candles recebidos para {ativo} (tf={timeframe})", level="info")

        # Salva CSV institucional (um arquivo por ativo)
        os.makedirs("dados", exist_ok=True)
        features_path = os.path.join("dados", f"features_{ativo}.csv")
        try:
            df.to_csv(features_path, index=False)
            log_event(f"[COLETOR] Features salvas em: {features_path}", level="info")
        except Exception as e:
            log_event(f"[COLETOR] Falha ao salvar CSV institucional: {e}", level="warning")

        # Sanity de colunas essenciais
        essenciais = ["timestamp", "open", "high", "low", "close", "tick_volume"]
        faltando = [col for col in essenciais if col not in df.columns]
        if faltando:
            log_event(f"[COLETOR] WARNING: Colunas faltando após coleta: {faltando}", level="warning")
            return pd.DataFrame()

        # Retorna somente as essenciais (ordem padronizada)
        return df[essenciais].copy()

    except Exception as e:
        msg = f"[COLETOR] ERRO: Exceção ao coletar candles: {e}"
        log_event(msg, level="error")
        try:
            enviar_telegram(msg)
        except Exception:
            pass
        return pd.DataFrame()
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass

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
