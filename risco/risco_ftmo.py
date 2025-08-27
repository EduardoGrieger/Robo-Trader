# -*- coding: utf-8 -*-
# Merged: risco_ftmo (original) + Fase 5 (Timezone & perda do dia consistentes)
from __future__ import annotations
from typing import Tuple, Dict, Any
import duckdb
import os
import json
from datetime import datetime, timedelta, timezone

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

# --- imports do projeto
from gestao.gestao_posicoes import risco_aberto_ftmo
from utils.debug_logger import log_event
# util novo de timezone (Fase 5)
try:
    from utils.time_utils import inicio_fim_dia_servidor_utc
except Exception:
    # fallback (comportamento antigo) caso util ainda não exista
    inicio_fim_dia_servidor_utc = None  # type: ignore

# Caminhos institucionais
base_path = os.path.dirname(os.path.dirname(__file__))
db_path = os.path.join(base_path, "dados", "robodados.duckdb")
config_path = os.path.join(base_path, "config.json")

# =========================
# Config & Helpers (originais)
# =========================
def carregar_config():
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_event(f"[RISCO] ERRO ao carregar config: {e}", level="error")
        return {}

def validar_tabela_operacoes():
    try:
        con = duckdb.connect(db_path)
        schema = con.execute("PRAGMA table_info('operacoes')").fetchall()
        colunas = [c[1] for c in schema]
        tipos = {c[1]: (c[2] or "").upper() for c in schema}

        # Adiciona coluna resultado se não existir
        if "resultado" not in colunas:
            log_event("[DB] Coluna 'resultado' não encontrada, criando...", level="warning")
            try:
                con.execute("ALTER TABLE operacoes ADD COLUMN resultado DOUBLE")
                log_event("[DB] Coluna 'resultado' criada com sucesso.", level="info")
            except Exception as e:
                log_event(f"[DB] Erro ao criar coluna resultado: {e}", level="error")

        # Aviso se timestamp não for TIMESTAMP (apenas informativo)
        if "timestamp" in tipos and "TIMESTAMP" not in tipos["timestamp"]:
            log_event(f"[DB] AVISO: coluna 'timestamp' é '{tipos['timestamp']}', as consultas farão CAST para TIMESTAMP.", level="warning")

        con.close()
    except Exception as e:
        log_event(f"[DB] Erro ao validar tabela operacoes: {e}", level="error")

# =========================
# Janela do dia (atualizada para Fase 5)
# =========================
def _janela_dia_utc(config: dict) -> tuple[datetime, datetime]:
    """
    Calcula início/fim do 'dia' usando timezone_servidor_offset_horas do config.
    Agora usa utils.time_utils.inicio_fim_dia_servidor_utc quando disponível.
    """
    try:
        offset_h = float(config.get("timezone_servidor_offset_horas", -3.0))
    except Exception:
        offset_h = -3.0
    if callable(inicio_fim_dia_servidor_utc):
        # usa util novo — base no timezone do broker
        return inicio_fim_dia_servidor_utc(None, offset_h)
    # fallback compatível com o comportamento anterior
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc + timedelta(hours=offset_h)
    start_local = datetime(year=now_local.year, month=now_local.month, day=now_local.day, tzinfo=timezone.utc)  # tz placeholder
    start_utc = start_local - timedelta(hours=offset_h)
    end_utc = start_utc + timedelta(days=1)
    return start_utc, end_utc

# =========================
# Drawdown diário (original) — FIX: cast de timestamp e datetimes "naive"
# =========================
def _max_drawdown_diario(con, start_utc: datetime, end_utc: datetime) -> float:
    """Calcula o maior rebaixamento (peak-to-trough) do PnL realizado no dia."""
    import pandas as pd

    # Evita TIMESTAMPTZ: usa datetimes sem tz na query
    s_naive = start_utc.replace(tzinfo=None) if start_utc.tzinfo else start_utc
    e_naive = end_utc.replace(tzinfo=None) if end_utc.tzinfo else end_utc

    q = con.execute(
        """
        SELECT 
            CAST(resultado AS DOUBLE) AS r,
            CAST(timestamp AS TIMESTAMP) AS ts
        FROM operacoes
        WHERE CAST(resultado AS DOUBLE) IS NOT NULL
          AND CAST(timestamp AS TIMESTAMP) >= ?
          AND CAST(timestamp AS TIMESTAMP) < ?
        ORDER BY ts ASC
        """,
        (s_naive, e_naive),
    )
    df = q.fetchdf()
    if df.empty:
        return 0.0
    c = df['r'].cumsum()
    roll_max = c.cummax()
    dd = (c - roll_max).min()  # negativo ou zero
    return abs(float(dd or 0.0))

# =========================
# Regras de risco FTMO (originais, preservadas) — FIX: cast + naive
# =========================
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
        MAX_DRAWDOWN_DIARIO = config.get("max_drawdown_diario", None)

        con = duckdb.connect(db_path)

        # 1. Perdas consecutivas (original)
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

        # 2. Perda total do dia (realizada) com timezone do broker (atualizado)
        inicio_utc, fim_utc = _janela_dia_utc(config)
        # Evita TIMESTAMPTZ na comparação
        s_naive = inicio_utc.replace(tzinfo=None) if inicio_utc.tzinfo else inicio_utc
        e_naive = fim_utc.replace(tzinfo=None) if fim_utc.tzinfo else fim_utc

        df_total = con.execute("""
            SELECT SUM(CAST(resultado AS DOUBLE)) AS perda_dia
            FROM operacoes
            WHERE CAST(resultado AS DOUBLE) IS NOT NULL
              AND CAST(timestamp AS TIMESTAMP) >= ?
              AND CAST(timestamp AS TIMESTAMP) < ?
        """, (s_naive, e_naive)).fetchone()
        perda_dia = (df_total[0] or 0.0)
        percentual_perda = abs(perda_dia) / float(CAPITAL_CONTA or 1.0)

        log_event(f"[RISCO] Perda diária (realizada) [{inicio_utc.isoformat()} ~ {fim_utc.isoformat()}]: {perda_dia:.2f} | {percentual_perda*100:.2f}%")

        # 2b. Drawdown diário (peak-to-trough no dia) — original com FIX
        dd_dia = _max_drawdown_diario(con, inicio_utc, fim_utc)
        perc_dd_dia = dd_dia / float(CAPITAL_CONTA or 1.0)
        log_event(f"[RISCO] Drawdown DIÁRIO: {dd_dia:.2f} | Percentual: {perc_dd_dia*100:.2f}%")

        # 3. Drawdown TOTAL acumulado (original)
        df_drawdown = con.execute("""
            SELECT SUM(CASE WHEN resultado < 0 THEN resultado ELSE 0 END) AS drawdown_total
            FROM operacoes
        """).fetchone()
        drawdown_total = abs(df_drawdown[0]) if df_drawdown[0] is not None else 0.0
        percentual_drawdown_total = drawdown_total / float(CAPITAL_CONTA or 1.0)
        log_event(f"[RISCO] Drawdown TOTAL: {drawdown_total:.2f} | Percentual: {percentual_drawdown_total*100:.2f}%")

        # 4. Risco aberto FTMO (soma do risco SL) — original
        risco_aberto = risco_aberto_ftmo(ativo)
        limite_risco = (EXPOSICAO_MAXIMA_PERCENTUAL / 100.0) * float(CAPITAL_CONTA or 0.0)
        log_event(f"[RISCO] Risco aberto (SL) em {ativo}: {risco_aberto:.2f} | Limite: {limite_risco:.2f}")

        con.close()

        # REGRAS INSTITUCIONAIS — Se qualquer uma for violada, retorna True (preservadas)
        if percentual_perda >= (float(LIMITE_PERDA_DIARIA or 0.0) / 100.0):
            log_event(f"[RISCO] Perda diária excedida: {percentual_perda*100:.2f}% (limite: {LIMITE_PERDA_DIARIA}%)", level="warning")
            return True

        if percentual_drawdown_total >= float(MAX_DRAWDOWN_TOTAL or 0.0):
            log_event(f"[RISCO] Drawdown TOTAL excedido: {percentual_drawdown_total*100:.2f}% (limite: {float(MAX_DRAWDOWN_TOTAL or 0.0)*100:.2f}%)", level="warning")
            return True

        if perdas_consecutivas:
            log_event(f"[RISCO] {MAX_PERDAS_CONSECUTIVAS} perdas consecutivas detectadas.", level="warning")
            return True

        if MAX_DRAWDOWN_DIARIO is not None and perc_dd_dia >= float(MAX_DRAWDOWN_DIARIO):
            log_event(f"[RISCO] Drawdown DIÁRIO excedido: {perc_dd_dia*100:.2f}% (limite: {float(MAX_DRAWDOWN_DIARIO)*100:.2f}%)", level="warning")
            return True

        if risco_aberto > limite_risco:
            log_event(f"[RISCO] Limite de risco FTMO (soma dos SLs) excedido para {ativo}!", level="warning")
            return True

        log_event(f"[RISCO] Nenhuma violação de risco detectada para {ativo}", level="info")
        return False

    except Exception as e:
        log_event(f"[RISCO] Erro ao verificar risco FTMO: {e}", level="error")
        return False

# =========================
# API pública (original)
# =========================
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

    risco_absoluto = float(capital_conta or 0.0) * (float(risco_por_trade or 0.0) / 100.0)
    distancia_stop = abs(float(preco_entrada) - float(stop_loss))
    if distancia_stop == 0:
        return 0.01  # mínimo operacional

    # Corrigido: risco por trade dividido pelo valor do pip do lote padrão (mantido)
    lote = risco_absoluto / (distancia_stop * 100000)
    lote = max(0.01, round(lote, 2))  # mínimo de 0.01
    return lote

# =========================
# (NOVO) Utilitários da Fase 5 – opcionais para decisões externas
# =========================
def _mt5_sum_deals_utc(start_utc: datetime, end_utc: datetime) -> tuple[float, int]:
    if mt5 is None:
        return 0.0, 0
    try:
        deals = mt5.history_deals_get(start_utc, end_utc) or []
    except Exception as e:
        log_event(f"[RISCO] history_deals_get falhou: {e}", level="warning")
        deals = []
    total = 0.0
    for d in deals:
        try:
            total += float(getattr(d, "profit", 0.0) or 0.0)
        except Exception:
            pass
    return total, len(deals)

def _mt5_pnl_aberto_atual() -> tuple[float, int]:
    if mt5 is None:
        return 0.0, 0
    try:
        poss = mt5.positions_get() or []
    except Exception as e:
        log_event(f"[RISCO] positions_get falhou: {e}", level="warning")
        return 0.0, 0
    total = 0.0
    for p in poss:
        try:
            total += float(getattr(p, "profit", 0.0) or 0.0)
        except Exception:
            pass
    return total, len(poss)

def perda_diaria_atual(cfg: dict | None = None) -> Dict[str, Any]:
    """
    Cálculo de perda diária com base no timezone do broker (Fase 5), via MT5.
    Mantém o original (DuckDB) em violou_regras_risco_ftmo; este util permite
    decisões externas considerando PnL aberto.
    """
    cfg = cfg or carregar_config()
    try:
        offset_h = float(cfg.get("timezone_servidor_offset_horas", -3.0))
    except Exception:
        offset_h = -3.0

    if callable(inicio_fim_dia_servidor_utc):
        start_utc, end_utc = inicio_fim_dia_servidor_utc(None, offset_h)
    else:
        # fallback: mesma lógica do _janela_dia_utc
        start_utc, end_utc = _janela_dia_utc(cfg)

    realizada, n_deals = _mt5_sum_deals_utc(start_utc, end_utc)
    aberto, n_pos = _mt5_pnl_aberto_atual()
    com_aberto = realizada + aberto

    out = {
        "start_utc": start_utc.isoformat(),
        "end_utc": end_utc.isoformat(),
        "realizada": round(realizada, 2),
        "pnl_aberto": round(aberto, 2),
        "com_aberto": round(com_aberto, 2),
        "n_deals": int(n_deals),
        "n_pos": int(n_pos),
    }
    return out

def limite_perda_diaria_valor(cfg: dict | None = None) -> float:
    cfg = cfg or carregar_config()
    try:
        capital = float(cfg.get("capital_conta", 0.0) or 0.0)
    except Exception:
        capital = 0.0

    # 1) percentual (limite_loss_dia_percentual)
    lim_pct = cfg.get("limite_loss_dia_percentual", None)
    try:
        lim_pct = float(lim_pct) if lim_pct is not None else None
    except Exception:
        lim_pct = None
    val_pct = (lim_pct or 0.0) / 100.0 * capital if lim_pct is not None else None

    # 2) valor absoluto meta_loss_dia (pode vir negativo no config)
    meta = cfg.get("meta_loss_dia", None)
    try:
        val_meta = abs(float(meta)) if meta is not None else None
    except Exception:
        val_meta = None

    candidatos = [v for v in (val_pct, val_meta) if isinstance(v, (int, float)) and v > 0]
    return float(min(candidatos)) if candidatos else 0.0

def excedeu_perda_diaria(cfg: dict | None = None, incluir_aberto: bool = True) -> tuple[bool, Dict[str, Any]]:
    """
    Verifica se excedeu a perda diária (Fase 5): retorna (bool, detalhes).
    Usa timezone do broker; calcula via MT5 deals + opcional PnL aberto.
    Não altera a lógica atual de violou_regras_risco_ftmo automaticamente.
    """
    cfg = cfg or carregar_config()
    perda = perda_diaria_atual(cfg)
    limite = limite_perda_diaria_valor(cfg)
    base = perda["com_aberto"] if incluir_aberto else perda["realizada"]
    excedeu = (limite > 0.0) and (base <= -limite)
    detalhe = {
        "perda": perda,
        "limite": round(limite, 2),
        "criterio": "com_aberto" if incluir_aberto else "realizada",
        "excedeu": bool(excedeu),
    }
    return bool(excedeu), detalhe
