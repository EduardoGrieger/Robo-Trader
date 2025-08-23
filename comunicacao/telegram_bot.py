# comunicacao/telegram_bot.py
# -*- coding: utf-8 -*-
"""
Bot de comandos do Telegram — compatível com o main_loop

Correções e melhorias:
- Aceita "/status" (além de "status") e adiciona ajuda melhor.
- Autorização mais permissiva quando não há chat_id nem allowed_users definidos.
- Reset de offset: função resetar_offset() para destravar polling.
- Fallback de envio direto para a API se enviar_telegram_alertas falhar.
- consultar_status_ftmo: corrige chaves dos dicionários de ordens (price_open/sl/type).
- Logs de diagnóstico discretos (sem expor token).

Dependências: requests, duckdb (opcional), utils.debug_logger, utils.utils.
"""

import os
import time
import requests
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

try:
    from utils.debug_logger import log_event, log_exception
except Exception:
    def log_event(msg, level="info", **kw):  # fallback silencioso
        print(f"[{level.upper()}] {msg}")
    def log_exception(msg, e, **kw):
        print(f"[ERROR] {msg}: {e}")

from utils.utils import carregar_config
try:
    from comunicacao.telegram_alertas import enviar_telegram as _enviar_telegram_alerta
except Exception:
    _enviar_telegram_alerta = None

# Arquivos locais de controle
OFFSET_FILE = "dados/telegram_offset.txt"
COMANDO_FILE = "dados/comando_telegram.txt"

# ========= Helpers de arquivo =========
def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _safe_write(path: str, content: str) -> None:
    _ensure_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)

# ========= Config / credenciais =========
def _get_cfg() -> dict:
    """
    Lê credenciais e knobs do Telegram de env ou config.json.
    Nunca loga token/segredo.
    """
    cfg = carregar_config()
    token = os.getenv("TELEGRAM_BOT_TOKEN") or cfg.get("telegram_token")
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or cfg.get("telegram_chat_id")
    base = cfg.get("telegram_endpoint", "https://api.telegram.org").rstrip("/")
    timeout = float(cfg.get("telegram_timeout_sec", 8))
    max_retries = int(cfg.get("telegram_max_retries", 2))
    poll_timeout = int(cfg.get("telegram_poll_timeout_sec", 5))  # long polling curto
    allowed_users = cfg.get("telegram_allowed_users") or []
    # normaliza allowed_users para lista de strings
    if isinstance(allowed_users, (int, float)):
        allowed_users = [str(int(allowed_users))]
    elif isinstance(allowed_users, str):
        allowed_users = [allowed_users.strip()]
    else:
        allowed_users = [str(x).strip() for x in allowed_users if str(x).strip()]

    return {
        "token": token,
        "chat_id": str(chat_id) if chat_id is not None else None,
        "base_url": base,
        "timeout": timeout,
        "max_retries": max_retries,
        "poll_timeout": poll_timeout,
        "allowed_users": allowed_users,  # vazio => se chat_id também não vier, libera geral
    }

# ========= API compatível (offset & comandos) =========
def obter_ultimo_offset() -> int:
    """Lê o último offset salvo para evitar reprocessar mensagens já lidas."""
    try:
        if not os.path.exists(OFFSET_FILE):
            return 0
        with open(OFFSET_FILE, "r", encoding="utf-8") as f:
            return int((f.read() or "0").strip())
    except Exception as e:
        log_event(f"[TELEGRAM] Erro ao ler offset: {e}", level="error", modulo="telegram_bot")
        return 0

def salvar_ultimo_offset(offset: int) -> None:
    """Salva o último offset para controle da leitura no Telegram."""
    try:
        _safe_write(OFFSET_FILE, str(int(offset)))
    except Exception as e:
        log_event(f"[TELEGRAM] Erro ao salvar offset: {e}", level="error", modulo="telegram_bot")

def resetar_offset() -> None:
    """Zera o offset para destravar a leitura de novas mensagens (caso esteja travado)."""
    try:
        _safe_write(OFFSET_FILE, "0")
        log_event("[TELEGRAM] Offset resetado para 0.", level="warning", modulo="telegram_bot")
    except Exception as e:
        log_event(f"[TELEGRAM] Falha ao resetar offset: {e}", level="error", modulo="telegram_bot")

def setar_comando_telegram(comando: str) -> None:
    """Salva o comando recebido para execução posterior no robô."""
    try:
        _safe_write(COMANDO_FILE, str(comando))
    except Exception as e:
        log_event(f"[TELEGRAM] Erro ao salvar comando: {e}", level="error", modulo="telegram_bot")

def ler_comando_telegram() -> Optional[str]:
    """Lê o comando salvo anteriormente."""
    try:
        if not os.path.exists(COMANDO_FILE):
            return None
        with open(COMANDO_FILE, "r", encoding="utf-8") as f:
            return f.read().strip() or None
    except Exception as e:
        log_event(f"[TELEGRAM] Erro ao ler comando: {e}", level="error", modulo="telegram_bot")
        return None

# ========= Envio =========
def _enviar_telegram_direto(mensagem: str, chat_id: Optional[str], cfg: dict) -> bool:
    """Fallback direto usando sendMessage da API do Telegram."""
    try:
        token = cfg.get("token")
        cid = chat_id or cfg.get("chat_id")
        if not token or not cid:
            return False
        url = f"{cfg['base_url']}/bot{token}/sendMessage"
        data = {"chat_id": cid, "text": mensagem}
        resp = requests.post(url, data=data, timeout=cfg["timeout"])
        ok = bool(resp.ok and resp.json().get("ok"))
        if not ok:
            log_event(f"[TELEGRAM] Fallback sendMessage falhou: http={resp.status_code}", level="warning", modulo="telegram_bot")
        return ok
    except Exception as e:
        log_event(f"[TELEGRAM] Erro no fallback direto: {e}", level="error", modulo="telegram_bot")
        return False

def enviar_telegram(mensagem: str, chat_id: Optional[str] = None) -> bool:
    """
    Wrapper compatível: tenta via comunicacao.telegram_alertas; se falhar, usa fallback direto.
    """
    try:
        if _enviar_telegram_alerta:
            ok = _enviar_telegram_alerta(mensagem, chat_id=chat_id)
            if ok:
                return True
    except Exception as e:
        log_exception("[TELEGRAM] Falha no wrapper enviar_telegram (primeira via)", e, modulo="telegram_bot")
    # Fallback direto
    cfg = _get_cfg()
    return _enviar_telegram_direto(mensagem, chat_id, cfg)

# ========= Consultas: performance e status =========
def consultar_performance() -> dict:
    """Consulta performance agregada no DuckDB (se existir)."""
    try:
        import duckdb
        with duckdb.connect("dados/robodados.duckdb") as con:
            res = con.execute("""
                SELECT 
                    COUNT(*) AS total,
                    SUM(CASE WHEN resultado > 0 THEN 1 ELSE 0 END) AS acertos,
                    SUM(CASE WHEN resultado < 0 THEN 1 ELSE 0 END) AS erros,
                    SUM(CASE WHEN resultado = 0 THEN 1 ELSE 0 END) AS neutros,
                    SUM(CAST(resultado AS DOUBLE)) AS lucro,
                    AVG(CAST(volume AS DOUBLE) * CAST(preco_abertura AS DOUBLE)) AS investimento_medio
                FROM operacoes
                WHERE resultado IS NOT NULL
            """).fetchone()
        total, acertos, erros, neutros, lucro, inv_med = res or (0, 0, 0, 0, 0.0, 0.0)
        return {
            "total": int(total or 0),
            "acertos": int(acertos or 0),
            "erros": int(erros or 0),
            "neutros": int(neutros or 0),
            "lucro": float(lucro or 0.0),
            "investimento_medio": float(inv_med or 0.0),
        }
    except Exception as e:
        log_event(f"[TELEGRAM] Erro na consulta performance: {e}", level="error", modulo="telegram_bot")
        return {"erro": True, "mensagem": str(e)}

def consultar_status_ftmo() -> str:
    """Monta e retorna o status completo do robô com base nas informações financeiras."""
    try:
        import duckdb
        from gestao.gestao_posicoes import (
            saldo_bruto, valor_investido, exposicao_ftmo,
            obter_ordens_abertas_mt5, lucro_fechado, lucro_aberto,
        )
        config = carregar_config()
        saldo_inicial = float(config.get("capital_conta", 50000))
        saldo_atual = float(saldo_bruto())
        lucro_realizado = float(lucro_fechado())
        ativos = config.get("ativos", ["EURUSD"])

        # soma de lucro aberto apenas nos ativos listados
        try:
            lucro_aberto_total = sum(float(lucro_aberto(a)) for a in ativos)
        except Exception:
            lucro_aberto_total = 0.0

        drawdown_atual = saldo_atual - saldo_inicial

        # perda do dia no banco, se existir
        try:
            with duckdb.connect("dados/robodados.duckdb") as con:
                hoje = datetime.now().strftime("%Y-%m-%d")
                perda_dia_row = con.execute("""
                    SELECT SUM(CAST(resultado AS DOUBLE))
                    FROM operacoes
                    WHERE resultado IS NOT NULL
                      AND DATE(timestamp) = ?
                """, (hoje,)).fetchone()
                perda_dia = float((perda_dia_row[0] if perda_dia_row else 0) or 0)
        except Exception:
            perda_dia = 0.0

        limite_perda_dia = -abs(saldo_inicial * (config.get("limite_loss_dia_percentual", 2.5) / 100))
        limite_drawdown = -abs(saldo_inicial * (config.get("max_drawdown_total", 0.10)))
        limite_exposicao = abs(saldo_inicial * (config.get("exposicao_maxima_percentual", 1.0) / 100))
        risco_por_trade = abs(saldo_inicial * (config.get("risco_por_trade_percentual", 0.2) / 100))

        bloco_ativos = []
        for ativo in ativos:
            ordens = obter_ordens_abertas_mt5(ativo)  # dicionários
            exposicao = exposicao_ftmo(ativo)
            pnl_aberto = 0.0
            try:
                pnl_aberto = float(lucro_aberto(ativo))
            except Exception:
                pass

            # Corrige chaves: aceita tanto 'price_open'/'sl'/'type' quanto versões pt
            risco_aberto = 0.0
            for o in ordens:
                try:
                    volume = float(o.get("volume", 0) or 0)
                    preco_abertura = float((o.get("preco_abertura") or o.get("price_open") or 0) or 0)
                    stop_loss = float((o.get("stop_loss") or o.get("sl") or 0) or 0)

                    tipo_str = str(o.get("tipo", "")).lower().strip()
                    tipo_int = o.get("type", None)
                    is_buy = None
                    if tipo_str in ("compra", "buy"):
                        is_buy = True
                    elif tipo_str in ("venda", "sell"):
                        is_buy = False
                    elif isinstance(tipo_int, int):
                        is_buy = (tipo_int == 0)  # 0=buy, 1=sell

                    if stop_loss > 0 and preco_abertura > 0 and volume > 0 and is_buy is not None:
                        cs = 100000.0  # aproximação
                        if is_buy:
                            risco = max(0.0, (preco_abertura - stop_loss) * volume * cs)
                        else:
                            risco = max(0.0, (stop_loss - preco_abertura) * volume * cs)
                        risco_aberto += risco
                except Exception:
                    continue

            bloco_ativos.append(
                f"🟦 {ativo}\n"
                f"  • Ordens abertas: {len(ordens)}\n"
                f"  • Exposição nominal: ${exposicao:,.2f}\n"
                f"  • Lucro/perda aberta: ${pnl_aberto:,.2f}\n"
                f"  • Risco aberto (até o stop): ${risco_aberto:,.2f}"
            )

        status = (
            f"📊 STATUS DO ROBÔ FTMO\n"
            f"──────────────────────────────\n"
            f"💵 Saldo atual: ${saldo_atual:,.2f}\n"
            f"🔒 Saldo inicial: ${saldo_inicial:,.2f}\n"
            f"💸 Lucro realizado (fechado): ${lucro_realizado:,.2f}\n"
            f"💡 Lucro/perda em aberto: ${lucro_aberto_total:,.2f}\n"
            f"📉 Drawdown atual: ${drawdown_atual:,.2f} ({(drawdown_atual/saldo_inicial)*100:.2f}%)\n"
            f"📆 Perda diária acumulada: ${perda_dia:,.2f} (Limite: ${limite_perda_dia:,.2f})\n"
            f"\n─────────\nAtivos em aberto:\n─────────\n"
            + ("\n\n".join(bloco_ativos) if bloco_ativos else "— sem posições —") +
            f"\n\n─────────\nLimites FTMO configurados:\n─────────\n"
            f"• Perda diária máxima: ${limite_perda_dia:,.2f}\n"
            f"• Drawdown máximo: ${limite_drawdown:,.2f}\n"
            f"• Exposição máxima por ativo: ${limite_exposicao:,.2f}\n"
            f"• Risco por trade: ${risco_por_trade:,.2f} ({config.get('risco_por_trade_percentual', 0.2):.2f}%)"
        )
        return status
    except Exception as e:
        log_event(f"[TELEGRAM] Erro ao consultar status FTMO: {e}", level="error", modulo="telegram_bot")
        return f"❌ Erro ao consultar status FTMO: {e}"

# ========= Polling de comandos =========
def _authorized(chat_id: str, cfg: dict) -> bool:
    """
    Se houver allowed_users, aceita se chat_id estiver na lista.
    Senão, se houver chat_id padrão, aceita apenas ele.
    Senão (sem restrições definidas), aceita qualquer chat.
    """
    if cfg["allowed_users"]:
        return chat_id in cfg["allowed_users"]
    if cfg["chat_id"]:
        return chat_id == cfg["chat_id"]
    return True  # sem restrições definidas

def _get_updates(token: str, offset: int, timeout: int, base_url: str, req_timeout: float) -> Tuple[bool, dict]:
    """
    Chama getUpdates com long polling curto.
    Retorna (ok, payload_json).
    """
    url = f"{base_url}/bot{token}/getUpdates"
    params = {"offset": offset + 1, "timeout": timeout, "allowed_updates": ["message"]}
    try:
        resp = requests.get(url, params=params, timeout=req_timeout)
    except requests.RequestException as e:
        return False, {"error": str(e)}
    try:
        data = resp.json()
    except Exception:
        return False, {"error": f"http_{resp.status_code}", "text": resp.text[:300]}
    return bool(data.get("ok")), data

def checar_comando() -> None:
    """
    Verifica mensagens novas no Telegram e grava COMANDO_FILE quando pertinente.
    Comandos suportados:
      - "status" / "/status" -> responde com status FTMO + performance
      - "fechar" / "fechar todas" / "/fechartodas" -> seta comando 'fechar_todas'
      - "pausar" / "/pausar" -> seta comando 'pausar'
      - "help" / "/help" -> ajuda básica
    """
    try:
        cfg = _get_cfg()
        if not cfg["token"]:
            log_event("[TELEGRAM] Token ausente — checagem de comandos desativada.", level="warning", modulo="telegram_bot")
            return

        offset = obter_ultimo_offset()
        ok, payload = _get_updates(cfg["token"], offset, cfg["poll_timeout"], cfg["base_url"], cfg["timeout"])

        if not ok or "result" not in payload:
            err = payload.get("error") or payload
            if err:
                log_event(f"[TELEGRAM] getUpdates falhou: {err}", level="warning", modulo="telegram_bot")
            return

        for update in payload["result"]:
            try:
                if "message" not in update:
                    continue

                mensagem = update["message"]
                texto = str(mensagem.get("text", "")).strip()
                chat_id = str(mensagem.get("chat", {}).get("id"))
                update_id = int(update.get("update_id", 0))

                # Salva offset assim que visto (para não reprocessar)
                salvar_ultimo_offset(update_id)

                # Filtra chat/usuário autorizado
                if not _authorized(chat_id, cfg):
                    log_event(f"[TELEGRAM] Mensagem ignorada de chat_id não autorizado: {chat_id}", level="warning", modulo="telegram_bot")
                    continue

                t = texto.lower()
                if t in ("help", "/help"):
                    enviar_telegram(
                        "🤖 Comandos disponíveis:\n"
                        "• status — mostra saldo, limites FTMO e performance\n"
                        "• fechar / fechar todas — fecha todas as ordens\n"
                        "• pausar — pausa o robô (exige lógica no main_loop)\n",
                        chat_id=chat_id
                    )
                    continue

                if t in ("status", "/status"):
                    log_event("[TELEGRAM] Comando STATUS recebido.", level="info", modulo="telegram_bot")
                    perf = consultar_performance()
                    status = consultar_status_ftmo()
                    if "erro" in perf:
                        msg = f"❌ Erro ao consultar performance: {perf.get('mensagem','')}\n\n{status}"
                    else:
                        msg = (
                            f"{status}\n"
                            f"\n─────────\n"
                            f"📈 Total operações: {perf['total']}\n"
                            f"✅ Acertos: {perf['acertos']} | ❌ Erros: {perf['erros']} | ➖ Neutros: {perf['neutros']}\n"
                            f"📌 Investimento médio: ${perf['investimento_medio']:,.2f}"
                        )
                    enviar_telegram(msg, chat_id=chat_id)
                    continue

                if t in ("fechar", "fechar todas", "/fechartodas"):
                    log_event("[TELEGRAM] Comando FECHAR_TODAS recebido.", level="warning", modulo="telegram_bot")
                    enviar_telegram("🚨 Fechamento de todas as ordens solicitado. Executando...", chat_id=chat_id)
                    setar_comando_telegram("fechar_todas")
                    continue

                if t in ("pausar", "/pausar"):
                    log_event("[TELEGRAM] Comando PAUSAR recebido.", level="warning", modulo="telegram_bot")
                    enviar_telegram("⏸️ Robô pausado (implementar lógica de pausa no main_loop).", chat_id=chat_id)
                    setar_comando_telegram("pausar")
                    continue

                # Mensagem livre: apenas ecoa orientação
                enviar_telegram("Comando não reconhecido. Envie 'help' para ajuda.", chat_id=chat_id)

            except Exception as e:
                log_exception("[TELEGRAM] Erro ao processar update", e, modulo="telegram_bot")
                # segue para o próximo update

    except Exception as e:
        log_exception("[TELEGRAM] Erro ao checar comandos", e, modulo="telegram_bot")

__all__ = [
    "checar_comando",
    "enviar_telegram",
    "setar_comando_telegram",
    "ler_comando_telegram",
    "consultar_performance",
    "consultar_status_ftmo",
    "obter_ultimo_offset",
    "salvar_ultimo_offset",
    "resetar_offset",
]