import duckdb
import os

db_path = os.path.join("dados", "robodados.duckdb")
os.makedirs("dados", exist_ok=True)

print(f"Conectando ao banco em: {db_path}")
con = duckdb.connect(db_path)

# Dicionário: coluna -> tipo SQL
colunas_necessarias = {
    "sinal": "VARCHAR",
    "tipo": "VARCHAR",
    "padrao": "VARCHAR",
    "regime": "VARCHAR",
    "contexto": "VARCHAR",
    "motivo_saida": "VARCHAR",
    "score_reforco": "FLOAT",
    "data_fechamento": "TIMESTAMP"
    # Adicione mais colunas e tipos conforme necessidade!
}

print("Verificando colunas necessárias na tabela 'operacoes'...")

try:
    result = con.execute("PRAGMA table_info('operacoes')").fetchall()
except Exception as e:
    print(f"[ERRO] Tabela 'operacoes' não existe. Crie-a antes de rodar esse script!")
    con.close()
    exit(1)

colunas_existentes = [col[1] for col in result]

for coluna, tipo in colunas_necessarias.items():
    if coluna not in colunas_existentes:
        print(f"Coluna '{coluna}' não encontrada. Criando coluna...")
        try:
            con.execute(f"ALTER TABLE operacoes ADD COLUMN {coluna} {tipo}")
            print(f"Coluna '{coluna}' criada com sucesso.")
        except Exception as e:
            print(f"Erro ao criar coluna '{coluna}': {e}")
    else:
        print(f"Coluna '{coluna}' já existe.")

print("\nColunas atuais da tabela 'operacoes':")
for col in con.execute("PRAGMA table_info('operacoes')").fetchall():
    print(f" - {col[1]} ({col[2]})")

con.close()
print("Conexão encerrada.")