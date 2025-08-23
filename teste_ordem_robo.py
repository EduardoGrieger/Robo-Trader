import time
import os
from datetime import datetime
from utils.utils import carregar_config
from utils.operacao_institucional import abrir_ordem_e_registrar  # <- Use este módulo!
from mt5.executar_ordem_mt5 import fechar_ordem
from gestao.gestao_posicoes import (
    obter_ordens_abertas,
    saldo_bruto,
    valor_investido,
)
from utils.debug_logger import log_event
import duckdb
from colorama import Fore, Style, init

init(autoreset=True)

ATIVO_TESTE = "EURUSD"
TIPO_TESTE = "compra"   # ou "venda"
VOLUME_TESTE = 0.01     # micro-lote

BANCO_DUCKDB = "dados/robodados.duckdb"

def saldo_info(ativo):
    try:
        saldo = saldo_bruto(ativo)
        investido = valor_investido(ativo)
        saldo = float(saldo) if saldo is not None and saldo != "" else 0.0
        investido = float(investido) if investido is not None and investido != "" else 0.0
        print(f"Saldo bruto: {saldo:.2f} | Valor investido: {investido:.2f}")
        return saldo, investido
    except Exception as e:
        print(Fore.RED + f"Erro ao obter saldo/investido: {e}" + Style.RESET_ALL)
        return 0.0, 0.0

def consultar_operacoes_db():
    try:
        con = duckdb.connect(BANCO_DUCKDB)
        df = con.execute("SELECT * FROM operacoes ORDER BY id DESC LIMIT 5").fetchdf()
        print(Fore.YELLOW + "\nÚltimas 5 operações na tabela operacoes:" + Style.RESET_ALL)
        print(df)
        con.close()
        return df
    except Exception as e:
        print(Fore.RED + f"Erro ao consultar DuckDB: {e}" + Style.RESET_ALL)
        return None

def main():
    print(Fore.CYAN + "\n==== TESTE DE ORDEM ROBO INSTITUCIONAL ====" + Style.RESET_ALL)
    print(f"Ativo de teste: {ATIVO_TESTE} | Tipo: {TIPO_TESTE} | Volume: {VOLUME_TESTE}\n")
    log_event(f"[TESTE] Início do teste de ordem - {ATIVO_TESTE} - {TIPO_TESTE} - {VOLUME_TESTE}", level="info")

    print(Fore.YELLOW + "1) Checando saldo e ordens abertas ANTES:" + Style.RESET_ALL)
    saldo_antes, investido_antes = saldo_info(ATIVO_TESTE)
    ordens_antes = obter_ordens_abertas(ATIVO_TESTE)
    print(f"Ordens abertas antes: {len(ordens_antes)}")
    consultar_operacoes_db()

    print(Fore.YELLOW + "\n2) Enviando ordem de TESTE pelo módulo institucional..." + Style.RESET_ALL)
    resultado = abrir_ordem_e_registrar(
        ativo=ATIVO_TESTE,
        tipo=TIPO_TESTE,
        volume=VOLUME_TESTE,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        contexto="",
        observacao="TESTE_AUTOMATICO"
    )
    ticket = resultado.get("order")
    retcode = resultado.get("retcode")
    print("Resultado do envio:", resultado)
    log_event(f"[TESTE] Ordem enviada | ticket: {ticket} | retcode: {retcode}", level="info")
    time.sleep(3)  # Espera para MT5 registrar

    print(Fore.YELLOW + "\n3) Checando saldo e ordens abertas DEPOIS da abertura:" + Style.RESET_ALL)
    saldo_depois, investido_depois = saldo_info(ATIVO_TESTE)
    ordens_depois = obter_ordens_abertas(ATIVO_TESTE)
    print(f"Ordens abertas depois: {len(ordens_depois)}")
    df_ops = consultar_operacoes_db()

    ordem_aberta = None
    if ticket:
        print("Tickets abertos no banco:", [str(o.get("ticket")) for o in ordens_depois])
        for o in ordens_depois:
            ticket_ = str(o.get("ticket")) if isinstance(o, dict) else str(o[15]) if len(o) > 15 else None
            if ticket_ is not None and ticket_ == str(ticket):
                ordem_aberta = o
                break
        if ordem_aberta:
            print(Fore.GREEN + f"Ordem de teste ABERTA com sucesso! Ticket: {ticket}" + Style.RESET_ALL)
        else:
            print(Fore.RED + f"Ordem de teste NÃO encontrada em abertas! Ticket: {ticket}" + Style.RESET_ALL)

    # Verifica e destaca a operação no banco
    if df_ops is not None and ticket:
        filtro = df_ops[df_ops["ticket"] == ticket]
        if not filtro.empty:
            print(Fore.GREEN + f"\n🟢 Operação de TESTE identificada no banco (ticket={ticket}):" + Style.RESET_ALL)
            print(filtro)
        else:
            print(Fore.RED + f"\n🔴 Operação de TESTE NÃO encontrada no banco (ticket={ticket})!" + Style.RESET_ALL)

    print(Fore.YELLOW + "\n4) Fechando ordem de teste..." + Style.RESET_ALL)
    if ticket:
        resultado_fech = fechar_ordem(ticket, ATIVO_TESTE)
        print("Resultado do fechamento:", resultado_fech)
        log_event(f"[TESTE] Fechamento de ordem | ticket: {ticket} | resultado: {resultado_fech}", level="info")
        time.sleep(3)
    else:
        print(Fore.RED + "Ticket da ordem de teste NÃO identificado, não será possível fechar!" + Style.RESET_ALL)

    print(Fore.YELLOW + "\n5) Checando saldo e ordens abertas DEPOIS do fechamento:" + Style.RESET_ALL)
    saldo_final, investido_final = saldo_info(ATIVO_TESTE)
    ordens_finais = obter_ordens_abertas(ATIVO_TESTE)
    print(f"Ordens abertas após fechamento: {len(ordens_finais)}")
    df_final = consultar_operacoes_db()

    if ticket and df_final is not None:
        con = duckdb.connect(BANCO_DUCKDB)
        try:
            con.execute(
                "UPDATE operacoes SET observacao = 'TESTE_AUTOMATICO' WHERE ticket = ?",
                (ticket,)
            )
            con.execute(
                "DELETE FROM operacoes WHERE ticket = ?",
                (ticket,)
            )
            print(Fore.YELLOW + f"[LIMPEZA] Operação de teste removida do banco (ticket={ticket})" + Style.RESET_ALL)
            log_event(f"[TESTE] Limpeza da operação de teste do banco | ticket: {ticket}", level="info")
        finally:
            con.close()

    print(Fore.CYAN + "\n==== RESUMO DO TESTE ====" + Style.RESET_ALL)
    print(f"Saldo antes: {saldo_antes:.2f} | depois: {saldo_depois:.2f} | final: {saldo_final:.2f}")
    print(f"Investido antes: {investido_antes:.2f} | depois: {investido_depois:.2f} | final: {investido_final:.2f}")
    print(f"Ordens abertas antes: {len(ordens_antes)} | depois: {len(ordens_depois)} | após fechamento: {len(ordens_finais)}")
    print(Fore.GREEN + "Teste completo! Confira logs, banco e seu MetaTrader para validação final." + Style.RESET_ALL)
    log_event("[TESTE] Fim do teste de ordem institucional", level="info")

if __name__ == "__main__":
    main()
