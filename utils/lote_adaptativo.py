import json
from utils.debug_logger import log_event

def carregar_config():
    with open("config.json", "r") as f:
        return json.load(f)

def calcular_lote_adaptativo(
    saldo=None,
    volatilidade=1.0,
    drawdown_pct=0.0,          # ATENÇÃO: aqui esperamos PERCENTUAL (ex.: 6.0 = 6%)
    meta_dia_atingida=False,
    performance_recente=0.0,
    contexto=None,
    score_conf=1.0,
    ativo="EURUSD",
    hora=None,
    return_all=False,          # Se True, retorna detalhes para painel/debug
    verbose=False              # Se True, printa tudo na tela
):
    """
    Calcula o lote adaptativo institucional para a próxima ordem.

    Correções:
      - NÃO usamos mais o 'volumes[ativo]' dentro do ajuste_final (evita dupla capagem).
      - Aplicamos piso (e opcionalmente teto) com base em 'config["volumes"][ativo]'.
        -> Se o calculado < config, usamos o do config e LOGAMOS.
        -> Se o calculado > config e 'usar_volume_config_como_teto=True', capamos e LOGAMOS.

    Observações:
      - drawdown_pct esperado em PERCENTUAL (ex.: 6.0 = 6%).
      - valor_ponto fixo (10.0) para FX padrão; ajuste se necessário por ativo.
    """
    log_event(
        f"[LOTE-DEBUG] Recebido para calcular lote: saldo={saldo}, volatilidade={volatilidade}, "
        f"drawdown_pct={drawdown_pct}, meta_dia_atingida={meta_dia_atingida}, performance_recente={performance_recente}, "
        f"score_conf={score_conf}, ativo={ativo}, hora={hora}",
        level="debug"
    )
    try:
        config = carregar_config()
        saldo = saldo if saldo is not None else config.get("capital_conta", 50000.0)
        risco_pct = float(config.get("risco_por_trade_percentual", 1.0))  # %
        stop_pts = int(config.get("sl_pips", 20))
        volume_padrao = float(config.get("volume_padrao", 0.01))

        # Valor configurado para o ativo (vamos tratar como PISO e, opcionalmente, TETO)
        volume_cfg = float(config.get("volumes", {}).get(ativo, volume_padrao))

        # Flags opcionais no config (defaults seguros)
        usar_volume_config_como_piso = bool(config.get("usar_volume_config_como_piso", True))
        usar_volume_config_como_teto = bool(config.get("usar_volume_config_como_teto", True))

        max_drawdown = float(config.get("max_drawdown_diario", 0.05))  # fração (ex.: 0.05 = 5%)

        # ------------------------------
        # 1) Risco absoluto em USD
        # ------------------------------
        risco_usd = float(saldo) * (risco_pct / 100.0) * float(score_conf)

        # ------------------------------
        # 2) Ajustes (não incluir 'volume_cfg' aqui para não capar em dobro)
        # ------------------------------
        # Volatilidade: se volatilidade for ~0.006, 1/0.006 ~ 166 → saturamos em 2.0 (mesmo do seu código)
        ajuste_volatilidade = max(0.2, min(2.0, 1.0 / max(0.01, float(volatilidade))))

        # Drawdown: drawdown_pct esperado em PERCENTUAL (6.0 = 6%)
        ajuste_drawdown = 1.0
        if drawdown_pct > 100.0 * max_drawdown:     # > 5% diário com default
            ajuste_drawdown = 0.7
        if drawdown_pct > 2 * 100.0 * max_drawdown: # > 10% diário com default
            ajuste_drawdown = 0.4
        if meta_dia_atingida:
            ajuste_drawdown *= 0.3
        if performance_recente < 0:
            ajuste_drawdown *= 0.7

        # ------------------------------
        # 3) Lote teórico
        # ------------------------------
        valor_ponto = 10.0  # padrão FX (ajuste se necessário)
        if stop_pts * valor_ponto == 0:
            lote_bruto = 0.01
        else:
            lote_bruto = risco_usd / (stop_pts * valor_ponto)

        # **Importante**: ajuste_final NÃO inclui volume_cfg (evita dupla capagem)
        ajuste_final = min(ajuste_volatilidade, ajuste_drawdown)
        lote_calculado = max(0.0, lote_bruto * ajuste_final)

        # ------------------------------
        # 4) Aplicar piso/teto do config
        # ------------------------------
        lote_utilizado = lote_calculado
        aplicou_piso = False
        aplicou_teto = False

        if usar_volume_config_como_piso and lote_utilizado < volume_cfg:
            lote_utilizado = volume_cfg
            aplicou_piso = True

        if usar_volume_config_como_teto and lote_utilizado > volume_cfg:
            lote_utilizado = volume_cfg
            aplicou_teto = True

        # Redondeamento final (duas casas, padrão do seu fluxo)
        lote_utilizado = round(lote_utilizado, 2)

        # ------------------------------
        # 5) Log detalhado
        # ------------------------------
        msg = (
            f"[LOTE] {ativo} | lote_teorico={lote_calculado:.4f} | lote_utilizado={lote_utilizado:.2f} | "
            f"volume_cfg={volume_cfg:.2f} | "
            f"ajustes=vol:{ajuste_volatilidade:.3f}, dd:{ajuste_drawdown:.3f}, score:{float(score_conf):.2f} | "
            f"saldo={float(saldo):.2f} | risco_pct={risco_pct:.2f} | stop_pts={stop_pts} | "
            f"valor_ponto={valor_ponto} | contexto={contexto}"
        )
        if aplicou_piso:
            msg += " | 🟰 Piso do config.json aplicado (lote elevado ao mínimo do ativo)"
        if aplicou_teto:
            msg += " | ⚠️ Teto do config.json aplicado (lote capado ao máximo do ativo)"
        log_event(msg, level="info")
        if verbose:
            print(msg)

        # ------------------------------
        # 6) Retorno
        # ------------------------------
        if return_all:
            return {
                "lote_bruto_teorico": round(lote_calculado, 4),
                "lote_utilizado": lote_utilizado,
                "volume_cfg": volume_cfg,
                "aplicou_piso": aplicou_piso,
                "aplicou_teto": aplicou_teto,
                "origem": {
                    "ajuste_volatilidade": ajuste_volatilidade,
                    "ajuste_drawdown": ajuste_drawdown,
                    "score_conf": float(score_conf),
                },
                "saldo": float(saldo),
                "risco_pct": float(risco_pct),
                "stop_pts": int(stop_pts),
                "valor_ponto": float(valor_ponto),
                "contexto": contexto
            }

        return lote_utilizado

    except Exception as e:
        log_event(f"[LOTE] Erro ao calcular lote para {ativo}: {e}", level="error")
        if verbose:
            print(f"[LOTE] ERRO ao calcular lote: {e}")
        return 0.01
