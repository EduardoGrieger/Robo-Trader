import duckdb
from datetime import datetime
from utils.debug_logger import log_event
from utils.utils import carregar_config

def atingiu_meta_periodo(db_path="dados/robodados.duckdb", tabela="operacoes"):
    """
    Checa se a meta mensal de gain ou a diária de loss foram atingidas.
    - Usa meta_gain_mes (% lucro alvo no mês, do config.json)
    - Usa meta_loss_dia (% perda máxima no dia, do config.json)
    - Usa capital_conta como saldo base
    Retorna: (atingiu: bool, tipo: str ["gain", "loss", ""], valor: float)
    """
    try:
        config = carregar_config()
        meta_gain_mes = config.get("meta_gain_mes", 10.0)
        meta_loss_dia = config.get("meta_loss_dia", -1.0)
        saldo_inicial = config.get("capital_conta", 10000)

        if saldo_inicial == 0:
            log_event("[META PERÍODO] Saldo inicial zero, impossível calcular meta.", level="warning")
            return False, "", 0.0

        hoje = datetime.now()
        mes_atual = hoje.strftime("%Y-%m")
        dia_atual = hoje.strftime("%Y-%m-%d")
        con = duckdb.connect(db_path)

        # Cálculo do gain MENSAL (corrigido para cast)
        res_mes = con.execute(f"""
            SELECT SUM(CAST(resultado AS DOUBLE))
            FROM {tabela}
            WHERE resultado IS NOT NULL
              AND strftime('%Y-%m', CAST(timestamp AS TIMESTAMP)) = ?
        """, (mes_atual,)).fetchone()
        lucro_mes = float(res_mes[0] or 0)
        percentual_mes = 100 * lucro_mes / saldo_inicial

        # Cálculo do loss DIÁRIO (corrigido para cast)
        res_dia = con.execute(f"""
            SELECT SUM(CAST(resultado AS DOUBLE))
            FROM {tabela}
            WHERE resultado IS NOT NULL
              AND DATE(CAST(timestamp AS TIMESTAMP)) = ?
        """, (dia_atual,)).fetchone()
        prejuizo_dia = float(res_dia[0] or 0)
        percentual_dia = 100 * prejuizo_dia / saldo_inicial

        # Verificação
        if percentual_mes >= meta_gain_mes:
            log_event(f"[META PERÍODO] Meta gain MENSAL atingida: {percentual_mes:.2f}% (meta={meta_gain_mes}%)", level="info")
            return True, "gain", percentual_mes
        if percentual_dia <= meta_loss_dia:
            log_event(f"[META PERÍODO] Meta loss DIÁRIA atingida: {percentual_dia:.2f}% (meta={meta_loss_dia}%)", level="warning")
            return True, "loss", percentual_dia

        log_event(f"[META PERÍODO] Nenhuma meta atingida. Mês: {percentual_mes:.2f}%, Dia: {percentual_dia:.2f}%", level="debug")
        return False, "", percentual_mes

    except Exception as e:
        log_event(f"[META PERÍODO] ERRO ao calcular metas: {e}", level="error")
        return False, "", 0.0
