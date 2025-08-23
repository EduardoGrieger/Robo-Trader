import duckdb
import pandas as pd
import os

# Caminhos
DB_PATH = "dados/robodados.duckdb"
CSV_PATH = "dados/ordens_salvas.csv"

# Garante que a pasta existe
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

# Conecta ao banco
con = duckdb.connect(DB_PATH)

# Lista tabelas
tables = con.execute("SHOW TABLES").fetchdf()

if "operacoes" not in tables["name"].values:
    print("❌ Tabela 'operacoes' não encontrada no banco!")
    print("Tabelas existentes:", tables["name"].tolist())
else:
    # Mostra estrutura da tabela
    info = con.execute("PRAGMA table_info('operacoes')").fetchdf()
    print("\nEstrutura da tabela operacoes:")
    print(info)

    # Busca as ordens, do mais recente ao mais antigo
    df = con.execute("SELECT * FROM operacoes ORDER BY timestamp DESC").fetchdf()

    if df.empty:
        print("⚠️ Nenhuma ordem encontrada na tabela 'operacoes'.")
    else:
        print(f"✅ Total de ordens salvas: {len(df)}")
        print(df.head(20))  # Exibe as 20 últimas ordens

        # Exporta para CSV (substitui se já existir)
        df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"📁 Ordens exportadas para: {CSV_PATH}")

con.close()
