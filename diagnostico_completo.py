# diagnostico_completo.py
import os
import pandas as pd
import duckdb
from colorama import Fore, Style, init
from utils.sinal_utils import normalizar_sinal, sinal_to_str

init(autoreset=True)

ARQUIVO_FEATURES = "dados/features.csv"
BANCO_DUCKDB = "dados/robodados.duckdb"
TABELA_OPERACOES = "operacoes"

def valida_features():
    print(Fore.CYAN + "\n==== DIAGNÓSTICO: FEATURES.CSV ====" + Style.RESET_ALL)
    if not os.path.exists(ARQUIVO_FEATURES):
        print(Fore.RED + f"Arquivo não encontrado: {ARQUIVO_FEATURES}" + Style.RESET_ALL)
        return False

    df = pd.read_csv(ARQUIVO_FEATURES)
    print(Fore.YELLOW + f"Linhas: {df.shape[0]}, Colunas: {df.shape[1]}" + Style.RESET_ALL)
    print(f"Colunas: {list(df.columns)}")

    # Checa colunas essenciais
    cols_essenciais = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'sma_20', 'sma_50', 'bb_high', 'bb_low', 'sinal']
    faltando = [col for col in cols_essenciais if col not in df.columns]
    if faltando:
        print(Fore.RED + f"Colunas ausentes: {faltando}" + Style.RESET_ALL)
        return False

    # Checa se sinal é int e só -1, 0 ou 1
    sinais_unicos = set(df['sinal'].unique())
    if not sinais_unicos.issubset({-1, 0, 1}):
        print(Fore.RED + f"Sinais fora do padrão: {sinais_unicos - {-1,0,1}}" + Style.RESET_ALL)
        return False
    else:
        print(Fore.GREEN + "Sinais OK: somente -1, 0, 1 presentes." + Style.RESET_ALL)

    # Checa NaN
    if df.isnull().sum().sum() > 0:
        print(Fore.RED + "Há valores NaN/missing nos features!" + Style.RESET_ALL)
    else:
        print(Fore.GREEN + "Sem valores faltantes nos features." + Style.RESET_ALL)

    print(Fore.GREEN + "Arquivo features.csv OK." + Style.RESET_ALL)
    return True

def valida_operacoes():
    print(Fore.CYAN + "\n==== DIAGNÓSTICO: TABELA OPERACOES (DuckDB) ====" + Style.RESET_ALL)
    if not os.path.exists(BANCO_DUCKDB):
        print(Fore.RED + f"Banco não encontrado: {BANCO_DUCKDB}" + Style.RESET_ALL)
        return False

    try:
        con = duckdb.connect(BANCO_DUCKDB)
        df = con.execute(f"SELECT * FROM {TABELA_OPERACOES}").fetchdf()
        print(Fore.YELLOW + f"Linhas: {df.shape[0]}, Colunas: {df.shape[1]}" + Style.RESET_ALL)
        print(f"Colunas: {list(df.columns)}")
        if 'sinal' not in df.columns:
            print(Fore.RED + "Coluna 'sinal' ausente!" + Style.RESET_ALL)
            return False
        # Checa tipo da coluna sinal (tem que ser int)
        tipo_sinal = str(df['sinal'].dtype)
        print(f"Tipo da coluna sinal: {tipo_sinal}")
        if not pd.api.types.is_integer_dtype(df['sinal']):
            print(Fore.RED + "Coluna 'sinal' NÃO é integer!" + Style.RESET_ALL)
            return False

        # Checa se só tem -1, 0, 1 (ou está vazia)
        sinais_unicos = set(df['sinal'].unique())
        if not sinais_unicos.issubset({-1, 0, 1}):
            print(Fore.RED + f"Sinais fora do padrão: {sinais_unicos - {-1,0,1}}" + Style.RESET_ALL)
            return False
        else:
            print(Fore.GREEN + "Sinais OK: somente -1, 0, 1 presentes." + Style.RESET_ALL)
        print(Fore.GREEN + "Tabela operacoes OK." + Style.RESET_ALL)
        con.close()
        return True
    except Exception as e:
        print(Fore.RED + f"Erro ao acessar DuckDB: {e}" + Style.RESET_ALL)
        return False

def resumo_final(res1, res2):
    print(Fore.CYAN + "\n==== RESUMO FINAL ====" + Style.RESET_ALL)
    if res1 and res2:
        print(Fore.GREEN + "ROBO PRONTO PARA RODAR 🚦" + Style.RESET_ALL)
    else:
        print(Fore.RED + "ERROS CRÍTICOS ENCONTRADOS! Revise as mensagens acima." + Style.RESET_ALL)

if __name__ == "__main__":
    res_feat = valida_features()
    res_op = valida_operacoes()
    resumo_final(res_feat, res_op)
