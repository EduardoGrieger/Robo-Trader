# utils/protecao_loss_flutuante.py

from gestao.gestao_posicoes import obter_ordens_abertas_mt5
from datetime import datetime
import os
from utils.debug_logger import log_event

BLOQUEIO_FILE = "dados/bloqueio_loss_flutuante.txt"

def checar_loss_flutuante(ativos, max_ordens_loss_aberto=3):
    """
    Checa se existem 'max_ordens_loss_aberto' ou mais ordens simultâneas abertas em prejuízo.
    Retorna: (atingiu: bool, qtd: int, tickets_loss: list)
    """
    ordens_loss = []
    for ativo in ativos:
        ordens = obter_ordens_abertas_mt5(ativo)
        for ordem in ordens:
            if float(ordem.get("profit", 0)) < 0:
                ticket = ordem.get("ticket", ordem.get("order", ordem.get("ticket")))
                ordens_loss.append(ticket)
    if len(ordens_loss) >= max_ordens_loss_aberto:
        log_event(f"[PROTEÇÃO LOSS FLUTUANTE] Atingido limite de {max_ordens_loss_aberto} ordens em prejuízo simultâneo! Tickets: {ordens_loss}", level="warning")
        return True, len(ordens_loss), ordens_loss
    else:
        log_event(f"[PROTEÇÃO LOSS FLUTUANTE] Ordens em prejuízo simultâneo: {len(ordens_loss)} (limite: {max_ordens_loss_aberto})", level="debug")
        return False, len(ordens_loss), ordens_loss

def bloquear_loss_flutuante():
    """
    Bloqueia o robô pelo resto do dia por loss flutuante.
    """
    hoje = datetime.now().strftime("%Y-%m-%d")
    try:
        with open(BLOQUEIO_FILE, "w") as f:
            f.write(hoje)
        log_event(f"[PROTEÇÃO LOSS FLUTUANTE] Bloqueio ativado até o fim do dia {hoje}.", level="warning")
    except Exception as e:
        log_event(f"[PROTEÇÃO LOSS FLUTUANTE] ERRO ao ativar bloqueio: {e}", level="error")

def esta_bloqueado_loss_flutuante():
    """
    Retorna True se o robô está bloqueado por loss flutuante, liberando automaticamente no dia seguinte.
    """
    if not os.path.exists(BLOQUEIO_FILE):
        return False
    try:
        with open(BLOQUEIO_FILE, "r") as f:
            data_bloqueio = f.read().strip()
        hoje = datetime.now().strftime("%Y-%m-%d")
        if data_bloqueio == hoje:
            log_event(f"[PROTEÇÃO LOSS FLUTUANTE] Robô bloqueado por loss flutuante no dia {hoje}.", level="info")
            return True
        else:
            try:
                os.remove(BLOQUEIO_FILE)
                log_event(f"[PROTEÇÃO LOSS FLUTUANTE] Bloqueio flutuante liberado automaticamente. Novo dia: {hoje}.", level="info")
            except Exception as e:
                log_event(f"[PROTEÇÃO LOSS FLUTUANTE] ERRO ao remover arquivo de bloqueio: {e}", level="warning")
            return False
    except Exception as e:
        log_event(f"[PROTEÇÃO LOSS FLUTUANTE] ERRO ao checar bloqueio: {e}", level="error")
        return False
