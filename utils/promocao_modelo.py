# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
import datetime
import json

# ------------------------------------------------------------
# Compat: funções originais (mantidas) — NÃO REMOVER
# ------------------------------------------------------------
def comparar_metricas(modelo_atual: Dict[str, Any], modelo_novo: Dict[str, Any], delta_min_acc_pp: float = 1.5) -> bool:
    """True se o novo superar o atual por >= delta_min_acc_pp em ACC ou F1_up/down (em pontos percentuais)."""
    if not modelo_atual:
        return True
    acc_old = float(modelo_atual.get("acc", 0.0)) * 100
    acc_new = float(modelo_novo.get("acc", 0.0)) * 100
    f1u_old = float(modelo_atual.get("f1_up", 0.0)) * 100
    f1u_new = float(modelo_novo.get("f1_up", 0.0)) * 100
    f1d_old = float(modelo_atual.get("f1_down", 0.0)) * 100
    f1d_new = float(modelo_novo.get("f1_down", 0.0)) * 100
    ganho_acc = acc_new - acc_old
    ganho_f1 = max(f1u_new - f1u_old, f1d_new - f1d_old)
    return (ganho_acc >= delta_min_acc_pp) or (ganho_f1 >= delta_min_acc_pp)

def gate_walkforward(veredito_wf: Dict[str, Any], pf_min: float = 1.2, acerto_min: float = 0.55, min_folds_ok: int = 3) -> bool:
    """
    Espera veredito_wf = { 'pf_por_fold': [...], 'acerto_por_fold': [...] }
    Aprova se pelo menos min_folds_ok folds satisfazem pf>=pf_min e acerto>=acerto_min.
    """
    if not isinstance(veredito_wf, dict):
        return False
    pf_folds = veredito_wf.get("pf_por_fold") or []
    acerto_folds = veredito_wf.get("acerto_por_fold") or []
    if not pf_folds or not acerto_folds:
        return False
    ok = 0
    for pf, ac in zip(pf_folds, acerto_folds):
        try:
            pf = float(pf)
            ac = float(ac)
        except Exception:
            continue
        if (pf >= float(pf_min)) and (ac >= float(acerto_min)):
            ok += 1
    return ok >= int(min_folds_ok)

# ------------------------------------------------------------
# Novos utilitários para promoção com telemetria (Fase 9)
# ------------------------------------------------------------
@dataclass
class Thresholds:
    delta_min_acc_pp: float = 1.5
    pf_min: float = 1.2
    acerto_min: float = 0.55
    min_folds_ok: int = 3

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]]) -> "Thresholds":
        if not isinstance(config, dict):
            return cls()
        ret = config.get("retreino", {}) if isinstance(config.get("retreino", {}), dict) else {}
        delta = float(ret.get("delta_min_promocao_acc_pp", cls.delta_min_acc_pp))
        gate = ret.get("gate_walkforward", {}) if isinstance(ret.get("gate_walkforward", {}), dict) else {}
        pf = float(gate.get("pf_min", cls.pf_min))
        ac = float(gate.get("acerto_min", cls.acerto_min))
        folds = int(gate.get("min_folds_ok", cls.min_folds_ok))
        return cls(delta_min_acc_pp=delta, pf_min=pf, acerto_min=ac, min_folds_ok=folds)

def _delta_pp(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(b - a) * 100.0

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)

def _agarrar_metricas(modelo: Dict[str, Any]) -> Dict[str, float]:
    """Extrai métricas-chave, tolerando ausências."""
    return {
        "acc": _safe_float(modelo.get("acc", 0.0)),
        "f1_up": _safe_float(modelo.get("f1_up", 0.0)),
        "f1_down": _safe_float(modelo.get("f1_down", 0.0)),
        # métricas adicionais opcionais
        "precision_up": _safe_float(modelo.get("precision_up", 0.0)),
        "precision_down": _safe_float(modelo.get("precision_down", 0.0)),
        "recall_up": _safe_float(modelo.get("recall_up", 0.0)),
        "recall_down": _safe_float(modelo.get("recall_down", 0.0)),
    }

def _contagem_folds_ok(veredito_wf: Dict[str, Any], th: Thresholds) -> int:
    """Conta quantos folds passam os thresholds (pf/acerto)."""
    pf = veredito_wf.get("pf_por_fold") or []
    ac = veredito_wf.get("acerto_por_fold") or []
    ok = 0
    for p, a in zip(pf, ac):
        try:
            if float(p) >= th.pf_min and float(a) >= th.acerto_min:
                ok += 1
        except Exception:
            continue
    return ok

def decidir_promocao(modelo_atual: Optional[Dict[str, Any]],
                     modelo_novo: Optional[Dict[str, Any]],
                     veredito_wf: Optional[Dict[str, Any]],
                     thresholds: Optional[Thresholds] = None,
                     salvar_telemetria_em: Optional[str] = None) -> Dict[str, Any]:
    """
    Combina comparação de métricas + gate WF em um único veredito explicável.
    Retorna payload com campos: aprovar, motivos[], deltas_pp{}, metricas_*, thresholds, wf{}, timestamp, [telemetria_path].
    """
    th = thresholds or Thresholds()
    m_old = _agarrar_metricas(modelo_atual or {})
    m_new = _agarrar_metricas(modelo_novo or {})

    delta_acc = _delta_pp(m_old["acc"], m_new["acc"])
    delta_f1u = _delta_pp(m_old["f1_up"], m_new["f1_up"])
    delta_f1d = _delta_pp(m_old["f1_down"], m_new["f1_down"])
    delta_f1_best = None
    if (delta_f1u is not None) or (delta_f1d is not None):
        vals = [v for v in [delta_f1u, delta_f1d] if v is not None]
        delta_f1_best = max(vals) if vals else None

    motivos: List[str] = []
    passa_delta = comparar_metricas(modelo_atual or {}, modelo_novo or {}, delta_min_acc_pp=th.delta_min_acc_pp)
    if not passa_delta:
        motivos.append(f"Δ mínimo não atingido (acc:+{(delta_acc or 0):.2f}pp, f1_best:+{(delta_f1_best or 0):.2f}pp; limiar={th.delta_min_acc_pp:.2f}pp)")

    wf_ok = gate_walkforward(veredito_wf or {}, pf_min=th.pf_min, acerto_min=th.acerto_min, min_folds_ok=th.min_folds_ok)
    folds_ok = _contagem_folds_ok(veredito_wf or {}, th)
    if not wf_ok:
        motivos.append(f"Gate WF reprovado (folds_ok={folds_ok}, exige >= {th.min_folds_ok}, pf_min={th.pf_min}, acerto_min={th.acerto_min})")

    aprovar = (passa_delta and wf_ok) or (not modelo_atual)  # se não há modelo atual, aprova

    payload = {
        "aprovar": bool(aprovar),
        "motivos": motivos,
        "deltas_pp": {
            "acc": delta_acc,
            "f1_up": delta_f1u,
            "f1_down": delta_f1d,
            "f1_best": delta_f1_best,
        },
        "metricas_atual": m_old,
        "metricas_novo": m_new,
        "thresholds": asdict(th),
        "wf": {
            "pf_por_fold": (veredito_wf or {}).get("pf_por_fold"),
            "acerto_por_fold": (veredito_wf or {}).get("acerto_por_fold"),
            "folds_ok": folds_ok,
        },
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

    # Telemetria opcional da decisão
    if salvar_telemetria_em:
        try:
            from pathlib import Path
            out_dir = Path(salvar_telemetria_em)
            out_dir.mkdir(parents=True, exist_ok=True)
            nome = f"promocao_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
            (out_dir / nome).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            payload["telemetria_path"] = str(out_dir / nome)
        except Exception as e:
            payload["telemetria_error"] = f"{e}"

    return payload
