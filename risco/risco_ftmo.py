import duckdb
import os
import json
from datetime import datetime
from gestao.gestao_posicoes import risco_aberto_ftmo
from utils.debug_logger import log_event

# Caminhos institucionais
base_path = os.path.dirname(os.path.dirname(__file__))
db_path = os.path.join(base_path, "dados", "robodados.duckdb")
config_path = os.path.join(base_path, "config.json")

def carregar_config():
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        log_event(f"[RISCO] ERRO ao carregar config: {e}", level="error")
        return {}

def validar_tabela_operacoes():
    try:
        con = duckdb.connect(db_path)
        schema = con.execute("PRAGMA table_info('operacoes')").fetchall()
        colunas = [c[1] for c in schema]
        # Adiciona coluna resultado se não existir
        if "resultado" not in colunas:
            log_event("[DB] Coluna 'resultado' não encontrada, criando...", level="warning")
            try:
                con.execute("ALTER TABLE operacoes ADD COLUMN resultado DOUBLE")
                log_event("[DB] Coluna 'resultado' criada com sucesso.", level="info")
            except Exception as e:
                log_event(f"[DB] Erro ao criar coluna resultado: {e}", level="error")
        con.close()
    except Exception as e:
        log_event(f"[DB] Erro ao validar tabela operacoes: {e}", level="error")

def violou_regras_risco_ftmo(ativo):
    validar_tabela_operacoes()  # Garante schema correto
    log_event("[RISCO] Iniciando verificação de regras de risco FTMO...")

    try:
        config = carregar_config()
        LIMITE_PERDA_DIARIA = config.get("limite_loss_dia_percentual", 2.5)  # % do saldo
        MAX_PERDAS_CONSECUTIVAS = config.get("limite_loss_dia_consecutivos", 3)
        CAPITAL_CONTA = config.get("capital_conta", 50000)
        EXPOSICAO_MAXIMA_PERCENTUAL = config.get("exposicao_maxima_percentual", 1.0)  # % por ativo
        MAX_DRAWDOWN_TOTAL = config.get("max_drawdown_total", 0.10)  # 10% do saldo

        con = duckdb.connect(db_path)

        # 1. Perdas consecutivas
        df = con.execute("""
            SELECT resultado
            FROM operacoes
            WHERE resultado IS NOT NULL
            ORDER BY id DESC
            LIMIT ?
        """, (MAX_PERDAS_CONSECUTIVAS,)).fetchdf()

        perdas_consecutivas = (
            all(r < 0 for r in df['resultado'])
            if len(df) == MAX_PERDAS_CONSECUTIVAS
            else False
        )

        # 2. Perda total do dia (lucro/prejuízo realizados no dia)
        df_total = con.execute("""
            SELECT SUM(CAST(resultado AS DOUBLE)) AS perda_dia
            FROM operacoes
            WHERE CAST(resultado AS DOUBLE) IS NOT NULL
              AND DATE(timestamp) = DATE(NOW())
        """).fetchone()
        perda_dia = df_total[0] or 0.0
        percentual_perda = abs(perda_dia) / CAPITAL_CONTA

        log_event(f"[RISCO] Perda diária: {perda_dia:.2f} | Percentual: {percentual_perda*100:.2f}%")

        # 3. Drawdown TOTAL acumulado (todas perdas históricas somadas)
        df_drawdown = con.execute("""
            SELECT SUM(CASE WHEN resultado < 0 THEN resultado ELSE 0 END) AS drawdown_total
            FROM operacoes
        """).fetchone()
        drawdown_total = abs(df_drawdown[0]) if df_drawdown[0] is not None else 0.0
        percentual_drawdown_total = drawdown_total / CAPITAL_CONTA
        log_event(f"[RISCO] Drawdown TOTAL: {drawdown_total:.2f} | Percentual: {percentual_drawdown_total*100:.2f}%")

        # 4. Risco aberto FTMO (soma do risco SL de todas as ordens abertas)
        risco_aberto = risco_aberto_ftmo(ativo)
        limite_risco = (EXPOSICAO_MAXIMA_PERCENTUAL / 100) * CAPITAL_CONTA
        log_event(f"[RISCO] Risco aberto (SL) em {ativo}: {risco_aberto:.2f} | Limite: {limite_risco:.2f}")

        con.close()

        # REGRAS INSTITUCIONAIS — Se qualquer uma for violada, retorna True
        if percentual_perda >= (LIMITE_PERDA_DIARIA / 100):
            log_event(f"[RISCO] Perda diária excedida: {percentual_perda*100:.2f}% (limite: {LIMITE_PERDA_DIARIA}%)", level="warning")
            return True

        if percentual_drawdown_total >= MAX_DRAWDOWN_TOTAL:
            log_event(f"[RISCO] Drawdown TOTAL excedido: {percentual_drawdown_total*100:.2f}% (limite: {MAX_DRAWDOWN_TOTAL*100:.2f}%)", level="warning")
            return True

        if perdas_consecutivas:
            log_event(f"[RISCO] {MAX_PERDAS_CONSECUTIVAS} perdas consecutivas detectadas.", level="warning")
            return True

        if risco_aberto > limite_risco:
            log_event(f"[RISCO] Limite de risco FTMO (soma dos SLs) excedido para {ativo}!", level="warning")
            return True

        log_event(f"[RISCO] Nenhuma violação de risco detectada para {ativo}", level="info")
        return False

    except Exception as e:
        log_event(f"[RISCO] Erro ao verificar risco FTMO: {e}", level="error")
        return False

# === Ponto Único de Decisão ===
def verificar_risco(ativo):
    """
    Função chamada pelo main_loop para decidir se pode operar.
    Só retorna True se todas as regras forem respeitadas!
    """
    log_event(f"[RISCO] Checando regras de risco FTMO para ativo: {ativo}", level="info")
    return not violou_regras_risco_ftmo(ativo)

def calcular_lote_dinamico(preco_entrada, stop_loss, capital_conta=None, risco_por_trade=None):
    """
    Calcula o lote a ser operado com base no risco por trade definido no config.
    Exemplo:
        risco_por_trade = 1.0 (%) -> risco máximo permitido do capital
    """
    config = carregar_config()
    if capital_conta is None:
        capital_conta = config.get("capital_conta", 50000)
    if risco_por_trade is None:
        risco_por_trade = config.get("risco_por_trade_percentual", 0.5)

    risco_absoluto = capital_conta * (risco_por_trade / 100)
    distancia_stop = abs(preco_entrada - stop_loss)
    if distancia_stop == 0:
        return 0.01  # mínimo operacional

    # Corrigido: risco por trade dividido pelo valor do pip do lote padrão
    lote = risco_absoluto / (distancia_stop * 100000)
    lote = max(0.01, round(lote, 2))  # mínimo de 0.01
    return lote
