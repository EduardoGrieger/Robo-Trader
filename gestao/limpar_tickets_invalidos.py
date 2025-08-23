import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import duckdb
from colorama import Fore, Style, init
from utils.debug_logger import log_event

init(autoreset=True)

# Garante que o diretório raiz esteja no sys.path para imports absolutos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def limpar_tickets_invalidos():
    """
    Remove registros da tabela 'operacoes' onde a coluna 'ticket' está zerada ou nula.
    """
    db_path = os.path.join("dados", "robodados.duckdb")
    if not os.path.exists(db_path):
        msg = f"[ERRO] Banco de dados não encontrado em {db_path}"
        print(Fore.RED + msg + Style.RESET_ALL)
        log_event(msg, level="error")
        return 0

    con = None
    try:
        con = duckdb.connect(db_path)

        # Confirma se coluna ticket existe
        colunas = [row[1] for row in con.execute("PRAGMA table_info('operacoes')").fetchall()]
        if "ticket" not in colunas:
            msg = "[ERRO] Coluna 'ticket' não encontrada na tabela 'operacoes'."
            print(Fore.RED + msg + Style.RESET_ALL)
            log_event(msg, level="error")
            return 0

        to_delete = con.execute(
            "SELECT COUNT(*) FROM operacoes WHERE ticket = 0 OR ticket IS NULL"
        ).fetchone()[0]

        if to_delete:
            msg = f"[GESTAO] Removendo {to_delete} registro(s) com ticket = 0 ou nulo..."
            print(Fore.YELLOW + msg + Style.RESET_ALL)
            log_event(msg, level="warning")

            con.execute("DELETE FROM operacoes WHERE ticket = 0 OR ticket IS NULL")

            msg = "[GESTAO] Limpeza de tickets inválidos concluída."
            print(Fore.GREEN + msg + Style.RESET_ALL)
            log_event(msg, level="info")
        else:
            msg = "[GESTAO] Nenhum registro inválido encontrado para limpeza."
            print(Fore.GREEN + msg + Style.RESET_ALL)
            log_event(msg, level="info")

        return to_delete

    except Exception as e:
        msg = f"[ERRO] Exceção ao limpar tickets inválidos: {e}"
        print(Fore.RED + msg + Style.RESET_ALL)
        log_event(msg, level="error")
        return 0

    finally:
        if con:
            con.close()

if __name__ == "__main__":
    limpar_tickets_invalidos()
