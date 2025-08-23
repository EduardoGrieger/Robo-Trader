from datetime import datetime
from utils.debug_logger import log_event

def dentro_horario_operacao(horarios_cfg):
    """
    Retorna True se agora está dentro de algum horário de operação definido.
    horarios_cfg: lista de dicts [{"inicio":"HH:MM", "fim":"HH:MM"}]
    Suporta horários cruzando a meia-noite (ex: {"inicio": "22:00", "fim": "02:00"})
    """
    agora = datetime.now().strftime("%H:%M")
    for janela in horarios_cfg:
        inicio = janela.get("inicio", "00:00")
        fim = janela.get("fim", "23:59")
        if inicio <= fim:
            if inicio <= agora <= fim:
                log_event(f"[HORÁRIO] Dentro da janela operacional ({inicio} - {fim}), agora: {agora}", level="debug")
                return True
        else:
            # Horários que cruzam a meia-noite (ex: 22:00 até 02:00)
            if agora >= inicio or agora <= fim:
                log_event(f"[HORÁRIO] Dentro da janela operacional cruzando meia-noite ({inicio} - {fim}), agora: {agora}", level="debug")
                return True
    log_event(f"[HORÁRIO] Fora de todas as janelas operacionais. Agora: {agora}", level="debug")
    return False
