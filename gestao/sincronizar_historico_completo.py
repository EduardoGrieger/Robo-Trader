import os
import sys
import duckdb
import MetaTrader5 as mt5
from datetime import datetime
from utils.debug_logger import log_event

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
db_path = os.path.join(base_path, "dados", "robodados.duckdb")

def conectar_banco():
    try:
        con = duckdb.connect(db_path)
        log_event("[SYNC HISTÓRICO] Banco conectado com sucesso.", level="info")
        return con
    except Exception as e:
        log_event(f"[SYNC HISTÓRICO] ERRO ao conectar banco: {e}", level="error", exc_info=sys.exc_info())
        return None

def conectar_mt5():
    if not mt5.initialize():
        log_event("[SYNC HISTÓRICO] ERRO ao conectar ao MetaTrader 5", level="error")
        return False
    log_event("[SYNC HISTÓRICO] MetaTrader 5 conectado com sucesso.", level="info")
    return True

def gerar_proximo_id(con):
    try:
        res = con.execute("SELECT MAX(id) FROM operacoes").fetchone()
        return (res[0] or 0) + 1
    except Exception as e:
        log_event(f"[SYNC HISTÓRICO] ERRO ao gerar próximo id: {e}", level="error", exc_info=sys.exc_info())
        return 1

def inserir_ou_atualizar_ordem_fechada(con, deal):
    try:
        # Campos obrigatórios (garantir correspondência com sua tabela 'operacoes')
        data_fechamento = datetime.fromtimestamp(deal.time).strftime('%Y-%m-%d %H:%M:%S')
        preco_fechamento = float(deal.price)
        lucro = float(deal.profit)

        # Tenta localizar ordem aberta pelo ticket
        res = con.execute("SELECT id FROM operacoes WHERE ticket = ?", (int(deal.ticket),)).fetchone()

        if res:
            # Atualiza ordem existente (considera fechada agora)
            con.execute("""
                UPDATE operacoes SET
                    preco_fechamento = ?,
                    lucro = ?,
                    data_fechamento = ?
                WHERE ticket = ?
            """, (
                preco_fechamento,
                lucro,
                data_fechamento,
                int(deal.ticket),
            ))
            log_event(f"[SYNC HISTÓRICO] Ordem fechada ATUALIZADA: ticket={deal.ticket}, lucro={deal.profit}", level="info")
        else:
            # Preenche os campos conforme sua tabela completa, os não disponíveis ficam None
            next_id = gerar_proximo_id(con)
            tipo_ordem = "compra" if getattr(deal, "type", 1) == 0 else "venda"
            preco_abertura = float(getattr(deal, "price_open", deal.price))
            con.execute("""
                INSERT INTO operacoes (
                    id, timestamp, ativo, padrao, regime, contexto, hora, tipo, volume,
                    preco_abertura, preco_fechamento, preco_saida, lucro, sinal, resultado,
                    ticket, retcode, motivo_fechamento, observacao, motivo_saida, score_reforco,
                    data_fechamento, preco_entrada, modelo_usado
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                next_id,
                data_fechamento,  # timestamp
                deal.symbol,
                "", "", "", "",  # padrao, regime, contexto, hora
                tipo_ordem,
                float(deal.volume),
                preco_abertura,
                preco_fechamento,
                None,   # preco_saida
                lucro,
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
            log_event(f"[SYNC HISTÓRICO] Ordem fechada INSERIDA: ticket={deal.ticket}, lucro={deal.profit}", level="info")
    except Exception as e:
        log_event(f"[SYNC HISTÓRICO] ERRO ao inserir/atualizar ordem fechada: {e}", level="error", exc_info=sys.exc_info())

def sincronizar_historico_completo():
    log_event("[SYNC HISTÓRICO] Iniciando sincronismo COMPLETO do histórico MT5...", level="info")
    con = conectar_banco()
    if not con:
        return
    if not conectar_mt5():
        con.close()
        return

    try:
        ano = datetime.now().year
        mes = datetime.now().month
        utc_from = datetime(ano, mes, 1)
        utc_to = datetime.now()
        deals = mt5.history_deals_get(utc_from, utc_to)

        if not deals:
            log_event("[SYNC HISTÓRICO] Nenhum deal encontrado no histórico MT5 nesse período!", level="warning")
            return

        novos = 0
        for deal in deals:
            inserir_ou_atualizar_ordem_fechada(con, deal)
            novos += 1

        log_event(f"[SYNC HISTÓRICO] Sincronismo COMPLETO concluído. {novos} ordens processadas.", level="info")
    except Exception as e:
        log_event(f"[SYNC HISTÓRICO] ERRO geral: {e}", level="error", exc_info=sys.exc_info())
    finally:
        con.close()
        mt5.shutdown()
        log_event("[SYNC HISTÓRICO] Conexão finalizada.", level="info")

if __name__ == "__main__":
    sincronizar_historico_completo()
