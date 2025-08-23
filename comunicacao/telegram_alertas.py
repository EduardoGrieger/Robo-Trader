# comunicacao/telegram_alertas.py
import os
import time
import json
import requests
from typing import Optional
from utils.debug_logger import log_event, log_exception
from utils.utils import carregar_config

# Limite do Telegram para mensagens de texto
_TELEGRAM_MAX_LEN = 4096
_ALLOWED_PARSE = {"Markdown", "MarkdownV2", "HTML"}

def _get_cfg():
    """
    Lê configurações do Telegram de:
      1) Variáveis de ambiente (prioridade)
      2) config.json
    """
    cfg = carregar_config()
    token = os.getenv("TELEGRAM_BOT_TOKEN") or cfg.get("telegram_token")
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or cfg.get("telegram_chat_id")
    base = cfg.get("telegram_endpoint", "https://api.telegram.org")
    disable_preview = bool(cfg.get("telegram_disable_web_preview", True))
    timeout = float(cfg.get("telegram_timeout_sec", 8))
    max_retries = int(cfg.get("telegram_max_retries", 2))
    return {
        "token": token,
        "chat_id": chat_id,
        "base_url": base.rstrip("/"),
        "disable_preview": disable_preview,
        "timeout": timeout,
        "max_retries": max_retries,
    }

def _truncate(msg: str) -> str:
    if msg is None:
        return ""
    s = str(msg)
    if len(s) <= _TELEGRAM_MAX_LEN:
        return s
    return s[:_TELEGRAM_MAX_LEN - 3] + "..."

def _validate_parse_mode(parse_mode: Optional[str]) -> Optional[str]:
    if not parse_mode:
        return None
    pm = str(parse_mode).strip()
    if pm not in _ALLOWED_PARSE:
        log_event(f"[TELEGRAM] parse_mode inválido '{parse_mode}' — ignorando.", level="warning", modulo="telegram")
        return None
    return pm

def enviar_telegram(mensagem: str,
                    chat_id: Optional[str] = None,
                    parse_mode: Optional[str] = None,
                    disable_notification: bool = False) -> bool:
    """
    Envia mensagem ao Telegram com robustez e segurança.

    Args:
        mensagem: Texto (será truncado em 4096 chars).
        chat_id:  Opcional; default vem do config/env.
        parse_mode: {"Markdown", "MarkdownV2", "HTML"} ou None.
        disable_notification: envia silenciosamente se True.

    Returns:
        bool: True se enviado com sucesso; False caso contrário.
    """
    try:
        cfg = _get_cfg()
        token = cfg["token"]
        default_chat = cfg["chat_id"]
        if not token or not default_chat:
            log_event("[TELEGRAM] Token ou Chat ID ausente(s). Configure telegram_token/telegram_chat_id ou variáveis de ambiente.", level="warning", modulo="telegram")
            return False

        url = f"{cfg['base_url']}/bot{token}/sendMessage"
        destino = chat_id or default_chat
        texto = _truncate(mensagem)
        pmode = _validate_parse_mode(parse_mode)
        payload = {
            "chat_id": destino,
            "text": texto,
            "disable_web_page_preview": cfg["disable_preview"],
            "disable_notification": bool(disable_notification),
        }
        if pmode:
            payload["parse_mode"] = pmode

        # Retries com tratamento de 429 (Too Many Requests)
        attempts = 0
        backoff = 1.0
        while True:
            attempts += 1
            try:
                resp = requests.post(url, json=payload, timeout=cfg["timeout"])
            except requests.RequestException as e:
                # Falha de rede — tenta novamente se houver retries sobrando
                if attempts <= cfg["max_retries"]:
                    log_event(f"[TELEGRAM] Falha de rede (tentativa {attempts}): {e}. Retentando em {backoff:.1f}s...", level="warning", modulo="telegram")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                log_exception("[TELEGRAM] Erro de rede definitivo ao enviar mensagem", e, modulo="telegram")
                return False

            if resp.ok:
                log_event(f"[TELEGRAM] Mensagem enviada para chat={destino}", level="info", modulo="telegram")
                return True

            # Trata respostas não-OK
            status = resp.status_code
            text = resp.text
            # Tenta extrair JSON com detalhes (e possivelmente retry_after)
            retry_after = None
            try:
                data = resp.json()
                # Exemplo de payload de erro: {"ok":false, "error_code":429, "parameters":{"retry_after": 12}, "description":"Too Many Requests: retry after 12"}
                params = data.get("parameters") or {}
                retry_after = params.get("retry_after")
            except Exception:
                pass

            if status == 429 and retry_after:
                # Respeita janela do Telegram
                wait_s = max(1, int(retry_after))
                log_event(f"[TELEGRAM] 429 Too Many Requests — aguardando {wait_s}s antes de retentar...", level="warning", modulo="telegram")
                time.sleep(wait_s)
                # Não conta contra o backoff normal; tenta de novo sem gastar retry
                continue

            # Outros erros HTTP — aplicar retry exponencial se possível
            if attempts <= cfg["max_retries"]:
                log_event(f"[TELEGRAM] HTTP {status} (tentativa {attempts}). Detalhes: {text[:300]}... Retentando em {backoff:.1f}s.", level="warning", modulo="telegram")
                time.sleep(backoff)
                backoff *= 2
                continue

            # Sem mais retries
            log_event(f"[ERRO TELEGRAM] HTTP {status}. Detalhes: {text[:500]}...", level="error", modulo="telegram")
            return False

    except Exception as e:
        log_exception("[TELEGRAM] Exceção inesperada no envio", e, modulo="telegram")
        return False
