import os
import sys
import duckdb
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from utils.debug_logger import log_event

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
db_path = os.path.join(base_path, "dados", "robodados.duckdb")

def conectar_banco():
    try:
        con = duckdb.connect(db_path)
        log_event("[SYNC INCREMENTAL] Banco conectado com sucesso.", level="info")
        return con
    except Exception as e:
        log_event(f"[SYNC INCREMENTAL] ERRO ao conectar banco: {e}", level="error", exc_info=sys.exc_info())
        return None

def conectar_mt5():
    if not mt5.initialize():
        log_event("[SYNC INCREMENTAL] ERRO ao conectar ao MetaTrader 5", level="error")
        return False
    log_event("[SYNC INCREMENTAL] MetaTrader 5 conectado com sucesso.", level="info")
    return True

def gerar_proximo_id(con):
    try:
        res = con.execute("SELECT MAX(id) FROM operacoes").fetchone()
        return (res[0] or 0) + 1
    except Exception as e:
        log_event(f"[SYNC INCREMENTAL] ERRO ao gerar próximo id: {e}", level="error", exc_info=sys.exc_info())
        return 1

def inserir_banco_ordem_fechada(con, deal):
    try:
        next_id = gerar_proximo_id(con)
        tipo_ordem = "compra" if getattr(deal, "type", 1) == 0 else "venda"
        preco_abertura = float(getattr(deal, "price_open", deal.price))
        data_fechamento = datetime.fromtimestamp(deal.time).strftime('%Y-%m-%d %H:%M:%S')
        con.execute("""
            INSERT INTO operacoes (
                id, timestamp, ativo, padrao, regime, contexto, hora, tipo, volume,
                preco_abertura, preco_fechamento, preco_saida, lucro, sinal, resultado,
                ticket, retcode, motivo_fechamento, observacao, motivo_saida, score_reforco,
                data_fechamento, preco_entrada, modelo_usado
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            next_id,
            data_fechamento,    # timestamp
            deal.symbol,
            "", "", "", "",     # padrao, regime, contexto, hora
            tipo_ordem,
            float(deal.volume),
            preco_abertura,
            float(deal.price),
            None,   # preco_saida
            float(deal.profit),
            0,      # sinal
            0,      # resultado
            int(deal.ticket),
            str(getattr(deal, "retcode", "")),
            "sync_mt5",   # motivo_fechamento
            "", "",  # observacao, motivo_saida
            None,   # score_reforco
            data_fechamento,
            preco_abertura,
            "",     # modelo_usado
        ))
        log_event(f"[SYNC INCREMENTAL] Ordem fechada sincronizada: ticket={deal.ticket}, ativo={deal.symbol}, volume={deal.volume}, lucro={deal.profit}", level="debug")
    except Exception as e:
        log_event(f"[SYNC INCREMENTAL] ERRO ao inserir ordem fechada: {e}", level="error", exc_info=sys.exc_info())

def sincronizar_historico_incremental(minutos=60):
    log_event(f"[SYNC INCREMENTAL] Iniciando sincronismo INCREMENTAL dos últimos {minutos} minutos...", level="info")
    con = conectar_banco()
    if not con:
        return
    if not conectar_mt5():
        con.close()
        return

    try:
        agora = datetime.now()
        utc_from = agora - timedelta(minutes=minutos)
        utc_to = agora
        deals = mt5.history_deals_get(utc_from, utc_to)

        if not deals:
            log_event("[SYNC INCREMENTAL] Nenhum deal encontrado no período recente!", level="warning")
            return

        tickets_banco = set()
        res = con.execute("SELECT ticket FROM operacoes").fetchall()
        if res:
            tickets_banco = set(int(x[0]) for x in res if x[0] is not None)

        novos = 0
        for deal in deals:
            if int(deal.ticket) not in tickets_banco:
                inserir_banco_ordem_fechada(con, deal)
                novos += 1

        log_event(f"[SYNC INCREMENTAL] Sincronismo incremental concluído. {novos} novas ordens adicionadas.", level="info")
    except Exception as e:
        log_event(f"[SYNC INCREMENTAL] ERRO geral: {e}", level="error", exc_info=sys.exc_info())
    finally:
        con.close()
        mt5.shutdown()
        log_event("[SYNC INCREMENTAL] Conexão finalizada.", level="info")

if __name__ == "__main__":
    sincronizar_historico_incremental(minutos=60)
