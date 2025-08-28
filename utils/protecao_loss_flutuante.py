# utils/protecao_loss_flutuante.py
# Proteção de "loss flutuante" com modos soft/hard e bloqueio diário.
# - soft: só BLOQUEIA novas entradas (não fecha posições)
# - hard: fecha TODAS as posições e BLOQUEIA novas entradas
# Obs: o fechamento em si continua a cargo do main_loop (chame fechar_todas_ordens)
from __future__ import annotations

import os
from datetime import datetime
from typing import List, Tuple, Any, Dict

# --- deps opcionais (fallbacks seguros) ---------------------------------------
try:
    from utils.utils import carregar_config
except Exception:
    def carregar_config(*args, **kwargs) -> dict:
        return {}

try:
    from utils.debug_logger import log_event
except Exception:
    def log_event(msg: str, level: str = "info", modulo: str = "protecao_loss_flutuante"):
        pass

try:
    # contrato esperado: obter_ordens_abertas_mt5(ativo) -> List[dict] (com 'ticket','profit',...)
    from gestao.gestao_posicoes import obter_ordens_abertas_mt5  # type: ignore
except Exception:
    def obter_ordens_abertas_mt5(ativo: str) -> List[Dict[str, Any]]:  # stub defensivo
        return []

# ------------------------------------------------------------------------------

BLOQUEIO_FILE = "dados/bloqueio_loss_flutuante.txt"

def _hoje() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def _garantir_dir_dados():
    d = os.path.dirname(BLOQUEIO_FILE) or "."
    os.makedirs(d, exist_ok=True)

def modo_acao_loss_flutuante() -> str:
    """
    Lê o modo no config:
      config['protecao_flutuante']['flutuante_modo'] ∈ {'soft','hard'}
    Fallback p/ legacy: retorna 'hard' se não definido.
    """
    try:
        cfg = carregar_config()
        modo = (cfg.get("protecao_flutuante", {}) or {}).get("flutuante_modo", "hard")
        modo = str(modo).strip().lower()
        return modo if modo in ("soft", "hard") else "hard"
    except Exception:
        return "hard"

def _max_ordens_cfg(default: int = 3) -> int:
    """
    Lê o máximo do config (novo ou legado).
    """
    try:
        cfg = carregar_config()
        novo = (cfg.get("protecao_flutuante", {}) or {}).get("max_ordens_loss_aberto")
        legado = cfg.get("max_ordens_loss_flutuante")
        val = novo if novo is not None else legado
        return int(val) if val is not None else int(default)
    except Exception:
        return int(default)

def checar_loss_flutuante(ativos: List[str], max_ordens_loss_aberto: int | None = None) -> Tuple[bool, int, List[Any]]:
    """
    Checa se existem 'max_ordens_loss_aberto' ou mais ordens simultâneas em prejuízo (profit < 0).
    Retorna: (atingiu: bool, qtd_loss: int, tickets_loss: list)
    """
    limite = _max_ordens_cfg() if max_ordens_loss_aberto is None else int(max_ordens_loss_aberto)
    ordens_loss: List[Any] = []

    try:
        for ativo in (ativos or []):
            for o in (obter_ordens_abertas_mt5(ativo) or []):
                try:
                    profit = float(o.get("profit", 0.0))
                except Exception:
                    continue
                if profit < 0.0:
                    ordens_loss.append(o.get("ticket", o.get("order", o)))
    except Exception as e:
        log_event(f"[PROTEÇÃO LOSS FLUTUANTE] Erro ao coletar ordens: {e}", level="error")
        # Em caso de erro na coleta, não travar automaticamente
        return False, 0, []

    qtd = len(ordens_loss)
    atingiu = qtd >= limite if limite > 0 else False
    if atingiu:
        log_event(f"[PROTEÇÃO LOSS FLUTUANTE] LIMITE atingido: {qtd} ordens negativas (limite={limite}).", level="warning")
    else:
        log_event(f"[PROTEÇÃO LOSS FLUTUANTE] OK: {qtd} ordens negativas (limite={limite}).", level="info")

    return bool(atingiu), int(qtd), ordens_loss

def bloquear_loss_flutuante() -> None:
    """
    Marca o BLOQUEIO para o dia corrente (libera automaticamente no dia seguinte).
    """
    try:
        _garantir_dir_dados()
        hoje = _hoje()
        with open(BLOQUEIO_FILE, "w", encoding="utf-8") as f:
            f.write(hoje)
        log_event(f"[PROTEÇÃO LOSS FLUTUANTE] Bloqueio ATIVADO até o fim do dia {hoje}.", level="warning")
    except Exception as e:
        log_event(f"[PROTEÇÃO LOSS FLUTUANTE] ERRO ao ativar bloqueio: {e}", level="error")

def esta_bloqueado_loss_flutuante() -> bool:
    """
    True se o robô está bloqueado por loss flutuante (arquivo do dia presente).
    Libera automaticamente em dia seguinte.
    """
    if not os.path.exists(BLOQUEIO_FILE):
        return False
    try:
        with open(BLOQUEIO_FILE, "r", encoding="utf-8") as f:
            data_bloqueio = (f.read() or "").strip()
        hoje = _hoje()
        if data_bloqueio == hoje:
            log_event(f"[PROTEÇÃO LOSS FLUTUANTE] Robô BLOQUEADO por loss flutuante (dia {hoje}).", level="info")
            return True
        # se mudou o dia, libera
        try:
            os.remove(BLOQUEIO_FILE)
            log_event(f"[PROTEÇÃO LOSS FLUTUANTE] Bloqueio liberado automaticamente (novo dia: {hoje}).", level="info")
        except Exception as e:
            log_event(f"[PROTEÇÃO LOSS FLUTUANTE] ERRO ao remover arquivo de bloqueio: {e}", level="warning")
        return False
    except Exception as e:
        log_event(f"[PROTEÇÃO LOSS FLUTUANTE] ERRO ao checar bloqueio: {e}", level="error")
        return False

# ----------------------------------------------------------------------
# Helpers opcionais para o main_loop (não fecham posições aqui)
# ----------------------------------------------------------------------
def coletar_tickets_loss(ativos: List[str]) -> List[Any]:
    """
    Conveniência: retorna apenas os tickets em prejuízo.
    """
    _, _, tickets = checar_loss_flutuante(ativos)
    return tickets

def precisa_acionar_bloqueio(ativos: List[str], max_ordens_loss_aberto: int | None = None) -> bool:
    """
    Atalho: True se atingiu limite de loss flutuante e ainda não está bloqueado hoje.
    """
    atingiu, _, _ = checar_loss_flutuante(ativos, max_ordens_loss_aberto)
    return bool(atingiu and not esta_bloqueado_loss_flutuante())
