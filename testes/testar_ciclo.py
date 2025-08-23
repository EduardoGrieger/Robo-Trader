import os
from datetime import datetime
from utils.utils import carregar_config
from mt5.coletar_candles_mt5 import coletar_candles
from features.gerar_features import calcular_features
from inteligencia.estrategia_ia import gerar_sinal
from utils.lote_adaptativo import calcular_lote_adaptativo
from risco.risco_ftmo import verificar_risco
from inteligencia.clusterizar_regimes import detectar_regime
from inteligencia.contexto import analisar_contexto
from utils.debug_logger import log_event

import MetaTrader5 as mt5
from colorama import Fore, Style, init
init(autoreset=True)

def testar_ciclo(ativo=None):
    config = carregar_config()
    ativo = ativo or config.get("ativos", ["EURUSD"])[0]
    janela_candles = config.get("janela_candles", 96)
    timeframes = config.get("timeframes", {})
    timeframe_str = timeframes.get(ativo, "M1")
    TIMEFRAME_MAP = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    timeframe = TIMEFRAME_MAP.get(timeframe_str, mt5.TIMEFRAME_M1)

    print(Fore.YELLOW + f"=== TESTE INSTITUCIONAL DO CICLO [{ativo}] ===" + Style.RESET_ALL)
    log_event(f"TESTE CICLO INSTITUCIONAL para {ativo}")

    # 1. Coletar candles
    candles = coletar_candles(ativo, quantidade=janela_candles, timeframe=timeframe)
    if candles is None or candles.empty:
        print(Fore.RED + f"[ERRO] Candles vazios para {ativo}. Aborte o teste." + Style.RESET_ALL)
        return
    print(Fore.GREEN + f"[OK] {len(candles)} candles coletados para {ativo}" + Style.RESET_ALL)

    # 2. Calcular features
    candles_feat = calcular_features(candles, config)
    if candles_feat is None or candles_feat.empty:
        print(Fore.RED + "[ERRO] Não foi possível gerar features." + Style.RESET_ALL)
        return
    print(Fore.GREEN + f"[OK] Features calculadas: {list(candles_feat.columns)}" + Style.RESET_ALL)

    # 3. Detectar regime de mercado
    regime = detectar_regime(candles_feat)
    print(Fore.CYAN + f"Regime detectado: {regime}" + Style.RESET_ALL)

    # 4. Analisar contexto
    contexto = analisar_contexto(candles_feat)
    if isinstance(contexto, dict):
        contexto['regime'] = regime
    print(Fore.CYAN + f"Contexto analisado: {contexto}" + Style.RESET_ALL)

    # 5. Gerar sinal IA
    df_sinal = gerar_sinal(candles_feat, ativo, contexto=contexto)
    print(Fore.CYAN + f"Sinal gerado:\n{df_sinal}" + Style.RESET_ALL)

    # 6. Calcular lote adaptativo
    saldo_atual = config.get("capital_conta", 50000)
    volatilidade = candles_feat["close"].std()
    volume_usado = calcular_lote_adaptativo(
        saldo=saldo_atual,
        volatilidade=volatilidade,
        contexto=contexto,
        ativo=ativo,
        hora=datetime.now().strftime("%H:%M:%S")
    )
    print(Fore.MAGENTA + f"Lote adaptativo calculado: {volume_usado}" + Style.RESET_ALL)

    # 7. Checar risco
    risco_ok = verificar_risco(ativo)
    if risco_ok:
        print(Fore.GREEN + "[OK] Risco institucional: PERMITIDO operar." + Style.RESET_ALL)
    else:
        print(Fore.RED + "[ERRO] Risco institucional: BLOQUEADO operar." + Style.RESET_ALL)

    print(Fore.YELLOW + "=== FIM DO CICLO DE TESTE ===" + Style.RESET_ALL)

if __name__ == "__main__":
    testar_ciclo()
