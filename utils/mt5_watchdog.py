
from __future__ import annotations
from typing import Optional
import os, json, time, math, random

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

from utils.debug_logger import log_event

# --- Telegram safe send ---
def _enviar_telegram_safe(msg: str) -> None:
    for mod_path in ("utils.telegram_utils", "utils.telegram_bot", "telegram_utils"):
        try:
            mod = __import__(mod_path, fromlist=["enviar_telegram"])
            enviar_telegram = getattr(mod, "enviar_telegram", None)
            if callable(enviar_telegram):
                try:
                    enviar_telegram(msg)
                except Exception:
                    pass
                return
        except Exception:
            continue
    try:
        from builtins import enviar_telegram  # type: ignore
        if callable(enviar_telegram):
            try:
                enviar_telegram(msg)  # type: ignore
            except Exception:
                pass
    except Exception:
        pass

def _carregar_config() -> dict:
    base = os.path.dirname(os.path.dirname(__file__))
    cfg_path = os.path.join(base, "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

class MT5Watchdog:
    def __init__(self):
        self.connected = False
        self.failures = 0
        self.backoff = 5.0  # seconds
        self.max_backoff = 300.0  # 5 minutes
        self.last_try = 0.0
        self.last_ok = 0.0
        self.logged_login = False  # avoid spam on steady-ok
        self.warned_down = False

    def _should_try(self) -> bool:
        now = time.time()
        return (now - self.last_try) >= self.backoff

    def _rand_jitter(self, base: float) -> float:
        # +/- 20% jitter
        j = base * (0.8 + 0.4 * random.random())
        return max(1.0, j)

    def _mark_failure(self):
        self.failures += 1
        self.backoff = min(self.max_backoff, self._rand_jitter(self.backoff * 2 if self.failures > 1 else self.backoff))
        self.connected = False
        self.last_try = time.time()

    def _mark_success(self):
        self.connected = True
        self.failures = 0
        self.backoff = 5.0
        self.last_ok = time.time()
        self.last_try = time.time()

    def ensure_connected(self, cfg: Optional[dict] = None) -> bool:
        if mt5 is None:
            log_event("[WATCHDOG] MT5 indisponível (módulo não importou).", level="error")
            return False

        # Already OK? quick ping occasionally
        if self.connected and (time.time() - self.last_ok) < 30:
            return True

        if not self._should_try():
            return self.connected

        try:
            # If already initialized, do a lightweight ping
            ti_ok = False
            try:
                ti = mt5.terminal_info()
                if ti is not None and getattr(ti, "trade_allowed", None) is not None:
                    ti_ok = True
            except Exception:
                ti_ok = False

            if not ti_ok:
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                ok = mt5.initialize()
            else:
                ok = True

            if not ok:
                self._mark_failure()
                if not self.warned_down:
                    _enviar_telegram_safe("⚠️ MT5 desconectado — tentando reconectar com backoff.")
                    self.warned_down = True
                log_event("[WATCHDOG] Falha ao inicializar MT5.", level="warning")
                return False

            # Simple ping (account_info)
            ai = mt5.account_info()
            if ai is None:
                self._mark_failure()
                if not self.warned_down:
                    _enviar_telegram_safe("⚠️ MT5 sem account_info — tentando reconectar com backoff.")
                    self.warned_down = True
                log_event("[WATCHDOG] account_info vazio.", level="warning")
                return False

            # Success
            first_time = not self.logged_login
            self._mark_success()
            if self.warned_down or first_time:
                _enviar_telegram_safe("✅ MT5 conectado e pronto.")
            self.warned_down = False
            self.logged_login = True
            return True

        except Exception as e:
            self._mark_failure()
            log_event(f"[WATCHDOG] Exceção ao conectar MT5: {e}", level="error")
            return False

    def ping(self) -> bool:
        if mt5 is None:
            return False
        try:
            ai = mt5.account_info()
            ok = ai is not None
            if ok:
                self._mark_success()
                return True
            else:
                self.connected = False
                return False
        except Exception:
            self.connected = False
            return False

_GLOBAL_WD = MT5Watchdog()

def garantir_conexao_mt5(intervalo_reconexao_min: float | int = 5) -> bool:
    """Mantém MT5 conectado; usa backoff & jitter internamente.
    intervalo_reconexao_min é aceito por compatibilidade, mas o watchdog se auto regula.
    """
    cfg = _carregar_config()
    ok = _GLOBAL_WD.ensure_connected(cfg)
    return ok
