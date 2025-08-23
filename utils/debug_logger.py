# utils/debug_logger.py
import os
import sys
import csv
import threading
import traceback
from datetime import datetime, timezone

# ====== Diretórios/constantes ======
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_MAX_SIZE = 10 * 1024 * 1024  # 10 MB
ROTATE_KEEP = 5

# Nível mínimo global (pode ser alterado em runtime com set_log_level)
LEVELS = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
    "critical": 50,
}

# Lê de env (ex.: set LOG_LEVEL=info) ou usa default "debug"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "debug").lower()
if LOG_LEVEL not in LEVELS:
    LOG_LEVEL = "debug"

# Echo mínimo no console (apenas erros/críticos por padrão)
CONSOLE_MIN_LEVEL = "error"

# Locks para escrita thread-safe
_lock = threading.Lock()
_delay_lock = threading.Lock()


# ====== Helpers de arquivo/rotação ======
def _log_filename(base: str) -> str:
    """Gera nome diário do arquivo de log."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"{base}_{today}.log")


def _rotate_log(path: str, keep: int = ROTATE_KEEP) -> None:
    """Rotaciona arquivo se ultrapassar tamanho limite, mantendo até KEEP versões."""
    try:
        if os.path.exists(path) and os.path.getsize(path) >= LOG_MAX_SIZE:
            for i in range(keep, 0, -1):
                src = f"{path}.{i - 1}" if i > 1 else path
                dst = f"{path}.{i}"
                if os.path.exists(src):
                    os.replace(src, dst)
            # zera o arquivo principal após rotação
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
    except Exception as e:
        # Em caso extremo, não deixa de logar por falha na rotação
        print(f"[{datetime.utcnow():%Y-%m-%d %H:%M:%S}] [ERROR] Falha na rotação do log '{path}': {e}")


# ====== API de configuração ======
def set_log_level(level: str) -> None:
    """
    Ajusta o nível mínimo global de logging em runtime.
    Ex.: set_log_level("info")
    """
    global LOG_LEVEL
    lvl = str(level).lower().strip()
    if lvl in LEVELS:
        LOG_LEVEL = lvl
        log_event(f"[LOGGER] Nível ajustado para '{LOG_LEVEL}'", level="info", modulo="logger")
    else:
        log_event(f"[LOGGER] Nível inválido '{level}'. Mantido '{LOG_LEVEL}'", level="warning", modulo="logger")


def get_log_level() -> str:
    """Retorna o nível mínimo atual."""
    return LOG_LEVEL


# ====== Núcleo de logging ======
def _should_log(level: str) -> bool:
    """Decide se deve logar baseado no nível configurado."""
    l = LEVELS.get(level.lower(), LEVELS["info"])
    base = LEVELS.get(LOG_LEVEL, LEVELS["debug"])
    return l >= base


def _should_echo_console(level: str) -> bool:
    """Decide se deve imprimir no console."""
    l = LEVELS.get(level.lower(), LEVELS["info"])
    base = LEVELS.get(CONSOLE_MIN_LEVEL, LEVELS["error"])
    return l >= base


def log_event(evento, level: str = "info", modulo: str | None = None,
              contexto: str | None = None, exc_info=None) -> None:
    """
    Loga uma linha no arquivo diário 'debug_YYYY-MM-DD.log' e replica para
    'erros_YYYY-MM-DD.log' quando nível for error/critical.

    Uso correto:
        log_event("Mensagem", level="warning", modulo="estrategia_ia")
    """
    try:
        # Filtra nível (aceita níveis inválidos como 'info')
        level = str(level).lower().strip()
        if level not in LEVELS:
            level = "info"
        if not _should_log(level):
            return

        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        process_name = os.path.basename(sys.argv[0]) or "-"
        pid = os.getpid()

        modulo_str = f"[{modulo}]" if modulo else ""
        contexto_str = f"[{contexto}]" if contexto else ""

        # Garantia de string limpa
        try:
            msg = str(evento)
        except Exception:
            msg = repr(evento)

        linha = f"[{timestamp}] [{level.upper()}] [PID:{pid}] [{process_name}] {modulo_str}{contexto_str} {msg}"

        if exc_info:
            # Aceita tupla (exc_type, exc, tb) ou exceção direta
            if isinstance(exc_info, BaseException):
                linha += "\n" + "".join(traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__))
            elif isinstance(exc_info, tuple):
                linha += "\n" + "".join(traceback.format_exception(*exc_info))
            else:
                linha += "\n" + str(exc_info)

        log_file = _log_filename("debug")
        error_log_file = _log_filename("erros")

        with _lock:
            _rotate_log(log_file)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(linha + "\n")

            if level in ("error", "critical"):
                _rotate_log(error_log_file)
                with open(error_log_file, "a", encoding="utf-8") as f:
                    f.write(linha + "\n")

        if _should_echo_console(level):
            print(linha)

    except Exception as e:
        # Última linha de defesa: não quebrar a aplicação por falha de log
        print(f"[{datetime.utcnow():%Y-%m-%d %H:%M:%S}] [ERROR] Falha ao gravar log: {e}")


def log_exception(msg: str, exc: BaseException, modulo: str | None = None,
                  contexto: str | None = None, level: str = "error") -> None:
    """Atalho para logar exceção com traceback."""
    log_event(msg, level=level, modulo=modulo, contexto=contexto, exc_info=exc)


def log_decisao(modulo: str, decisao: str, motivo: str, contexto: str | None = None) -> None:
    """Registro padrão para decisões do robô."""
    evento = f"[DECISÃO] {decisao} | Motivo: {motivo}"
    log_event(evento, level="info", modulo=modulo, contexto=contexto)


# ====== Monitor de delay de execução ======
DELAY_CSV = os.path.join(LOG_DIR, "monitor_delay_execucao.csv")


def _to_utc(dt):
    """
    Garante que o datetime seja tz-aware em UTC.
    Se tz-naive, assume UTC.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def log_delay_execucao(timestamp_inicio_candle, timestamp_execucao_ordem,
                       ativo: str, ciclo: int, info_extra: str = "") -> float | None:
    """
    Registra delay (s) entre início do candle e execução da ordem.
    Também loga uma linha padrão no arquivo de log.
    """
    try:
        # Parse e normalização dos timestamps para UTC tz-aware
        if isinstance(timestamp_inicio_candle, str):
            t1 = datetime.fromisoformat(timestamp_inicio_candle)
        else:
            t1 = timestamp_inicio_candle

        if isinstance(timestamp_execucao_ordem, str):
            t2 = datetime.fromisoformat(timestamp_execucao_ordem)
        else:
            t2 = timestamp_execucao_ordem

        t1 = _to_utc(t1)
        t2 = _to_utc(t2)

        delay = (t2 - t1).total_seconds()
        linha = f"{t1.isoformat()},{t2.isoformat()},{delay:.3f},{ativo},{ciclo},{info_extra}\n"

        with _delay_lock:
            novo_arquivo = not os.path.exists(DELAY_CSV)
            with open(DELAY_CSV, "a", encoding="utf-8") as f:
                if novo_arquivo:
                    f.write("inicio_candle,execucao_ordem,delay_segundos,ativo,ciclo,info\n")
                f.write(linha)

        log_event(
            f"[MONITOR DELAY] Delay execução: {delay:.3f}s | Ativo: {ativo} | Ciclo: {ciclo} | {info_extra}",
            level="info",
            modulo="monitor_delay_execucao"
        )
        return delay

    except Exception as e:
        log_exception("[MONITOR DELAY] ERRO ao registrar delay", e, modulo="monitor_delay_execucao")
        return None


# ====== Snapshots úteis (último ciclo / último retreino) ======
def gravar_ultimo_ciclo_log(dados: dict) -> None:
    """
    Grava snapshot do ciclo mais recente em logs/ultimo_ciclo.csv.
    Sobrescreve a cada ciclo. Formato amigável para Excel.
    """
    if not dados:
        return
    arquivo = os.path.join(LOG_DIR, "ultimo_ciclo.csv")
    try:
        with open(arquivo, "w", encoding="utf-8", newline="") as f:
            # Se for tudo escalar, grava linha única; se houver dicts, expande cabeçalhos
            if all(not isinstance(v, dict) for v in dados.values()):
                writer = csv.DictWriter(f, fieldnames=dados.keys())
                writer.writeheader()
                writer.writerow(dados)
            else:
                header = set()
                for v in dados.values():
                    if isinstance(v, dict):
                        header.update(v.keys())
                header = sorted(header)
                writer = csv.writer(f)
                writer.writerow(["topico"] + header)
                for k, v in dados.items():
                    if isinstance(v, dict):
                        linha = [k] + [v.get(h, "") for h in header]
                        writer.writerow(linha)
                    else:
                        writer.writerow([k, v])
    except Exception as e:
        log_exception("[LOG CICLO] ERRO ao gravar ultimo_ciclo.csv", e)


def gravar_ultimo_retreino_log(dados: dict) -> None:
    """
    Grava snapshot do último retreinamento em logs/ultimo_retreino.csv.
    Sobrescreve a cada retreino. Formato amigável para auditoria.
    """
    if not dados:
        return
    arquivo = os.path.join(LOG_DIR, "ultimo_retreino.csv")
    try:
        with open(arquivo, "w", encoding="utf-8", newline="") as f:
            if all(not isinstance(v, dict) for v in dados.values()):
                writer = csv.DictWriter(f, fieldnames=dados.keys())
                writer.writeheader()
                writer.writerow(dados)
            else:
                header = set()
                for v in dados.values():
                    if isinstance(v, dict):
                        header.update(v.keys())
                header = sorted(header)
                writer = csv.writer(f)
                writer.writerow(["topico"] + header)
                for k, v in dados.items():
                    if isinstance(v, dict):
                        linha = [k] + [v.get(h, "") for h in header]
                        writer.writerow(linha)
                    else:
                        writer.writerow([k, v])
    except Exception as e:
        log_exception("[LOG RETREINO] ERRO ao gravar ultimo_retreino.csv", e)


__all__ = [
    "log_event",
    "log_exception",
    "log_decisao",
    "log_delay_execucao",
    "gravar_ultimo_ciclo_log",
    "gravar_ultimo_retreino_log",
    "set_log_level",
    "get_log_level",
]
