from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from typing import Dict, Deque, Tuple, Optional, List
from datetime import datetime, timezone
import json, os

@dataclass
class BiasParams:
    janela: int = 500                   # tamanho da janela de votos
    warn_threshold: float = 0.80        # % para caracterizar viés (ex.: 0.80 => 80%)
    min_window_for_warn: int = 200      # mínimo de amostras não-neutras para considerar o alerta
    reduzir_lote_quando_enviesado: bool = True
    mult_enviesado: float = 0.50        # multiplicador de lote sob viés (se habilitado)
    persist_path: Optional[str] = None  # caminho opcional para persistir estado

    @classmethod
    def from_config(cls, config: Dict) -> "BiasParams":
        cfg = (config or {}).get("vies", {}) if isinstance(config, dict) else {}
        return cls(
            janela = int(cfg.get("vote_monitor_janela", cls.janela)),
            warn_threshold = float(cfg.get("warn_threshold", cls.warn_threshold)),
            min_window_for_warn = int(cfg.get("min_window_for_warn", cls.min_window_for_warn)),
            reduzir_lote_quando_enviesado = bool(cfg.get("reduzir_lote_quando_enviesado", cls.reduzir_lote_quando_enviesado)),
            mult_enviesado = float(cfg.get("mult_enviesado", cls.mult_enviesado)),
            persist_path = cfg.get("persist_path")  # opcional
        )

class VoteMonitor:
    """
    Monitora viés direcional (-1/0/1) por ativo em janela deslizante.
    -1: short, 0: neutro, 1: long
    """
    def __init__(self, params: BiasParams):
        self.p = params
        self._q: Dict[str, Deque[int]] = defaultdict(lambda: deque(maxlen=self.p.janela))
        self._last_warn: Dict[str, str] = {}  # ativo -> ISO timestamp última notificação
        # persistência opcional
        self._persist_path = self.p.persist_path or os.path.join(os.getcwd(), "dados", "vote_monitor.json")
        os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
        self._load()

    # ----------------- persistência -----------------
    def _load(self) -> None:
        try:
            if not os.path.exists(self._persist_path):
                return
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            janela = int(data.get("_janela") or self.p.janela)
            if janela != self.p.janela:
                # se mudou o tamanho, recria deques
                self._q = defaultdict(lambda: deque(maxlen=self.p.janela))
            for ativo, votes in (data.get("series") or {}).items():
                dq = deque(votes[-self.p.janela:], maxlen=self.p.janela)
                self._q[ativo] = dq
            self._last_warn = data.get("last_warn") or {}
        except Exception:
            # ignora erros de carga
            pass

    def _save(self) -> None:
        try:
            serial = {k: list(v) for k, v in self._q.items()}
            data = {
                "_janela": self.p.janela,
                "series": serial,
                "last_warn": self._last_warn
            }
            tmp = self._persist_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self._persist_path)
        except Exception:
            pass

    # ----------------- API pública -----------------
    def add_vote(self, ativo: str, voto: int) -> None:
        """Registra um voto (-1, 0, 1) para o ativo."""
        try:
            v = int(voto)
        except Exception:
            return
        if v not in (-1, 0, 1):
            return
        self._q[ativo].append(v)
        self._save()

    def stats(self, ativo: str) -> Dict[str, float | int | str | None]:
        dq = self._q.get(ativo, deque(maxlen=self.p.janela))
        n = len(dq)
        if n == 0:
            return {"n_total": 0, "n_up": 0, "n_down": 0, "n_neutro": 0,
                    "ratio_up": 0.0, "ratio_down": 0.0, "bias": None, "bias_ratio": 0.0}
        n_up = sum(1 for x in dq if x == 1)
        n_down = sum(1 for x in dq if x == -1)
        n_neutro = sum(1 for x in dq if x == 0)
        efetivo = max(1, n_up + n_down)  # considera só direcionais para ratio
        ratio_up = n_up / efetivo
        ratio_down = n_down / efetivo
        bias_ratio = max(ratio_up, ratio_down)
        bias = "up" if ratio_up > ratio_down else ("down" if ratio_down > ratio_up else None)
        return {
            "n_total": n,
            "n_up": n_up,
            "n_down": n_down,
            "n_neutro": n_neutro,
            "ratio_up": ratio_up,
            "ratio_down": ratio_down,
            "bias": bias,
            "bias_ratio": bias_ratio,
        }

    def should_warn(self, ativo: str) -> bool:
        st = self.stats(ativo)
        efetivo = st["n_up"] + st["n_down"]
        if efetivo < self.p.min_window_for_warn:
            return False
        return float(st["bias_ratio"]) >= float(self.p.warn_threshold)

    def lot_multiplier(self, ativo: str) -> float:
        """Sugere multiplicador de lote sob viés. 1.0 se desabilitado ou sem viés extremo."""
        if not self.p.reduzir_lote_quando_enviesado:
            return 1.0
        return self.p.mult_enviesado if self.should_warn(ativo) else 1.0

    def mark_warn_sent(self, ativo: str) -> None:
        self._last_warn[ativo] = datetime.now(timezone.utc).isoformat()
        self._save()

    def last_warn(self, ativo: str) -> Optional[str]:
        return self._last_warn.get(ativo)

    def summary_all(self) -> Dict[str, Dict]:
        return {a: self.stats(a) for a in list(self._q.keys())}

# ------------- helpers de integração -------------

_singleton: Optional[VoteMonitor] = None

def get_vote_monitor(config: Optional[Dict] = None) -> VoteMonitor:
    """Singleton com auto-carregamento de config.json quando não for passado."""
    global _singleton
    if _singleton is None:
        cfg = config
        if cfg is None:
            try:
                from utils.utils import carregar_config
                cfg = carregar_config()
            except Exception:
                cfg = {}
        params = BiasParams.from_config(cfg or {})
        _singleton = VoteMonitor(params)
    return _singleton

def registrar_voto(ativo: str, voto: int, config: Optional[Dict] = None) -> None:
    """Uso simples: chame após computar o voto do ensemble/modelo (-1/0/1)."""
    vm = get_vote_monitor(config)
    vm.add_vote(ativo, voto)

def obter_multiplicador_lote_por_vies(ativo: str, config: Optional[Dict] = None) -> float:
    vm = get_vote_monitor(config)
    return vm.lot_multiplier(ativo)

# Aliases com/sem acento para compatibilidade
def resumo_vies(config: Optional[Dict] = None) -> Dict[str, Dict]:
    vm = get_vote_monitor(config)
    return vm.summary_all()

def resumo_viés(config: Optional[Dict] = None) -> Dict[str, Dict]:
    return resumo_vies(config)

def mark_warn(ativo: str) -> None:
    get_vote_monitor(None).mark_warn_sent(ativo)
