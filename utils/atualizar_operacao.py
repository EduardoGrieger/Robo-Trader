import duckdb
import os
from datetime import datetime
from colorama import Fore, Style

# Caminho institucional para o banco
base_path = os.path.dirname(os.path.dirname(__file__))
db_path = os.path.join(base_path, "dados", "robodados.duckdb")

def atualizar_operacao(ticket, preco_fechamento, lucro):
    """
    Atualiza uma operação existente no banco com preço de fechamento e lucro.

    :param ticket: ID da operação (ticket/retcode)
    :param preco_fechamento: preço de saída da operação
    :param lucro: lucro/prejuízo da operação (float)
    """
    try:
        con = duckdb.connect(db_path)
        # Atualiza preco_fechamento e lucro da operação pelo ticket
        con.execute("""
            UPDATE operacoes
            SET preco_fechamento = ?,
                lucro = ?,
                resultado = ?
            WHERE ticket = ?
        """, (preco_fechamento, lucro, lucro, ticket))
        con.close()
        print(Fore.GREEN + f"[OPERACAO] Operação {ticket} atualizada: preco_fechamento={preco_fechamento}, lucro={lucro}" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"[OPERACAO] Erro ao atualizar operação {ticket}: {e}" + Style.RESET_ALL)
