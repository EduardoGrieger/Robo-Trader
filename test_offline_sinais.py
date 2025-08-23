import argparse, os, pandas as pd
from utils.utils import carregar_config
from features.gerar_features import calcular_features
from inteligencia.estrategia_ia import gerar_sinal

def carregar_csv(caminho):
    # Tenta detectar separador automaticamente
    try:
        df = pd.read_csv(caminho, sep=None, engine="python")
    except Exception:
        try:
            df = pd.read_csv(caminho, sep=";")
        except Exception:
            df = pd.read_csv(caminho)

    # Normaliza nomes/colunas mínimas
    if "timestamp" not in df.columns and "time" in df.columns:
        df["timestamp"] = df["time"]
    if "tick_volume" not in df.columns:
        df["tick_volume"] = df.get("volume", 0)

    cols_ok = [c for c in ["timestamp","open","high","low","close","tick_volume"] if c in df.columns]
    if len(cols_ok) < 5:
        raise ValueError(f"CSV sem colunas mínimas. Encontrei: {list(df.columns)}")
    return df[cols_ok].copy()

def walk_forward(features, ativo, start=200):
    resultados = []
    for i in range(max(start,1), len(features)):
        sub = features.iloc[:i+1].copy()
        saida = gerar_sinal(sub, ativo, contexto=None) or {}
        resultados.append({
            "timestamp": sub.iloc[-1].get("timestamp",""),
            "sinal": saida.get("sinal", 0),
            "padrao": saida.get("padrao",""),
            "confianca": saida.get("confianca",""),
            "motivo": saida.get("motivo","")
        })
    return pd.DataFrame(resultados)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Caminho do CSV de candles")
    ap.add_argument("--ativo", default="EURUSD")
    ap.add_argument("--start", type=int, default=200, help="índice inicial do walk-forward")
    args = ap.parse_args()

    os.makedirs("dados", exist_ok=True)
    cfg = carregar_config()

    candles = carregar_csv(args.csv)
    print(f"[INFO] CSV carregado: {len(candles)} candles")

    feats = calcular_features(candles, cfg, ativo=args.ativo)
    if feats is None or feats.empty:
        print("Features vazias. Verifique o CSV.")
        return
    print(f"[INFO] Features calculadas: {len(feats)} linhas")

    # Ajuste dinâmico do start para evitar DF vazio
    max_start = max(1, len(feats) - 2)
    start_eff = min(max(args.start, 1), max_start)
    if start_eff != args.start:
        print(f"[WARN] start {args.start} ajustado para {start_eff} (dataset curto)")
    print(f"[INFO] Walk-forward iniciando em i={start_eff}")

    df_res = walk_forward(feats, args.ativo, start=start_eff)
    out = "dados/resultados_offline.csv"
    df_res.to_csv(out, index=False)
    print(f"OK! Resultados salvos em {out} (linhas: {len(df_res)})")

    if not df_res.empty and "sinal" in df_res.columns:
        dist = df_res["sinal"].value_counts(normalize=True).to_dict()
        print("Distribuição de sinais:", dist)
    else:
        print("Sem resultados suficientes para calcular distribuição (dataset curto ou erro upstream).")

if __name__ == "__main__":
    main()
