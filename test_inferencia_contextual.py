# test_inferencia_contextual.py
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from inteligencia.inferencia_contextual import prever_contexto

np.random.seed(42)

def make_candles(
    n=240, s0=1.10,
    mu=0.0, sigma=0.00025,
    vol_base=1000, last_vol_mult=1.0,
    spike_start=None, spike_sigma=None
):
    """
    Gera candles sintéticos com retornos ~ N(mu, sigma).
    - Se spike_start/spike_sigma forem dados, injeta um regime de alta vol no final.
    - Para cenário lateral puro (mu=0 e sem spike), remove drift acumulado.
    """
    rets = np.random.normal(mu, sigma, size=n)

    # Força "lateral" a ser realmente flat (sem drift) quando não há spike
    if mu == 0.0 and spike_start is None:
        rets = rets - rets.mean()

    # Injeta explosão de volatilidade perto do fim, se solicitado
    if spike_start is not None and spike_sigma is not None:
        rets[spike_start:] = np.random.normal(mu, spike_sigma, size=n - spike_start)

    price = s0 * np.cumprod(1.0 + rets)

    # timestamps minuto a minuto (UTC) — usa "min" (substitui o 'T' deprecado)
    t0 = pd.Timestamp.utcnow().floor("min") - pd.Timedelta(minutes=n)
    ts = pd.date_range(t0, periods=n, freq="min", tz="UTC")

    # volume com ruído + ajuste no último candle
    vol = (vol_base * (1 + 0.05 * np.random.randn(n))).clip(min=1)
    vol[-1] *= last_vol_mult

    df = pd.DataFrame({
        "timestamp": ts,
        "close": price,
        "tick_volume": vol.astype(int)
    })
    return df

def run_and_print(df, nome, salvar_csv=True):
    if salvar_csv:
        os.makedirs("dados", exist_ok=True)
        path = f"dados/candles_sinteticos_{nome}.csv"
        df.to_csv(path, index=False)
    ctx = prever_contexto(df, ativo=nome.upper())
    print(f"\n=== {nome.upper()} ===")
    print(f"horario={ctx['horario']}  vol={ctx['volatilidade']}  volume={ctx['volume']}  "
          f"tendencia={ctx['tendencia']}  squeeze={ctx['squeeze']}")
    return ctx

def main():
    # 1) LATERAL: baixa vol, sem drift (sigma menor e drift removido acima)
    df_lateral = make_candles(
        n=240, s0=1.10, mu=0.0, sigma=0.00015,
        vol_base=1000, last_vol_mult=1.0
    )

    # 2) TENDÊNCIA: viés positivo + volume final mais alto
    df_tend = make_candles(
        n=240, s0=1.00, mu=0.00040, sigma=0.00030,
        vol_base=1100, last_vol_mult=2.0
    )

    # 3) EXPLOSÃO: vol baixa e, nos últimos ~15 min, spike forte de vol + volume alto
    df_explo = make_candles(
        n=240, s0=1.05, mu=0.0, sigma=0.00025,
        vol_base=900, last_vol_mult=3.0,
        spike_start=240 - 15,  # explosão perto do fim p/ destacar std atual vs referência
        spike_sigma=0.0040
    )

    run_and_print(df_lateral, "lateral")
    run_and_print(df_tend, "tendencia")
    run_and_print(df_explo, "explosao")

if __name__ == "__main__":
    main()
