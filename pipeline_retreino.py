"""
Pipeline completo de retreino do robô IA FTMO
USO: python pipeline_retreino.py
"""

import sys
import os
import re
import json
import glob
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

PROJETO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJETO_ROOT)  # Garante execução sempre na raiz

sys.path.insert(0, PROJETO_ROOT)

# === IMPORTS ORIGINAIS ===
from comunicacao.telegram_bot import enviar_telegram
from utils.debug_logger import log_event
from utils.utils import carregar_config  # <— já existia no seu original

PYTHON_EXEC = sys.executable

LOG_DIR = os.path.join(PROJETO_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "debug_pipeline.log")

def enviar_alerta_telegram(mensagem: str):
    try:
        enviar_telegram(mensagem)
    except Exception as e:
        log_event(f"Falha ao enviar mensagem Telegram: {e}", level="error")

def rodar_script(path, descricao=None):
    script_path = os.path.normpath(os.path.join(PROJETO_ROOT, path))
    log_event(f"Rodando: {script_path} {'- ' + descricao if descricao else ''}", level="info")
    try:
        resultado = subprocess.run(
            [PYTHON_EXEC, script_path],
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace"
        )
        log_event(f"OK - {path}\n{resultado.stdout}", level="info")
        enviar_alerta_telegram(f"✅ Sucesso: {descricao or path}")
        return True
    except subprocess.CalledProcessError as e:
        erro_msg = f"❌ FALHA em {path}!\nSaída:\n{e.stdout}\nErro:\n{e.stderr}"
        log_event(erro_msg, level="error")
        enviar_alerta_telegram(erro_msg)
        return False

# ========= PROMOÇÃO/BACKUP controlados (só quando TROCAR) =========

def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _listar_mais_recente(padrao):
    arquivos = glob.glob(os.path.join(PROJETO_ROOT, padrao))
    if not arquivos:
        return None
    arquivos = sorted(arquivos, key=lambda p: os.path.getmtime(p), reverse=True)
    return arquivos[0]

def _backup(caminho):
    if not os.path.exists(caminho):
        return None
    base, ext = os.path.splitext(caminho)
    destino = f"{base}_backup_{_ts()}{ext}"
    shutil.copy2(caminho, destino)
    log_event(f"[RETREINO] Backup criado: {destino}", level="info")
    return destino

def _promover(canonical_path, candidate_path):
    """
    Faz backup do atual e promove o candidato para o caminho canônico.
    """
    _backup(canonical_path)
    os.makedirs(os.path.dirname(canonical_path), exist_ok=True)
    shutil.copy2(candidate_path, canonical_path)
    log_event(f"[RETREINO] PROMOVIDO: {candidate_path} -> {canonical_path}", level="info")
    enviar_alerta_telegram(f"🤖 Modelo promovido: {os.path.basename(canonical_path)}")

def _carregar_avaliacao():
    """
    Tenta ler um arquivo de avaliação gerado pelo treino para decidir promoção.
    Suporta múltiplos nomes comuns. Se não existir, retorna {}.
    """
    candidatos = [
        os.path.join(PROJETO_ROOT, "modelos", "avaliacao_ensemble.json"),
        os.path.join(PROJETO_ROOT, "logs", "avaliacao_ensemble.json"),
        os.path.join(PROJETO_ROOT, "logs", "metrics_ensemble.json"),
    ]
    for caminho in candidatos:
        if os.path.exists(caminho):
            try:
                with open(caminho, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
            except Exception as e:
                log_event(f"[RETREINO] Erro lendo {caminho}: {e}", level="warning")
    return {}

def _flag_novo_modelo_aprovado():
    """
    Alternativa simples: se existir um arquivo-flag, considera aprovados.
    """
    flag_path = os.path.join(PROJETO_ROOT, "modelos", "novo_modelo_aprovado.flag")
    return os.path.exists(flag_path)

def promover_modelos_se_aprovados():
    """
    Promove modelos APENAS quando aprovados pelo retreino.
    Critérios:
      1) Há um arquivo de avaliação com campos indicando melhora? (preferido)
      2) Ou existe um arquivo-flag 'novo_modelo_aprovado.flag'?
    Se nenhum existir, não promove (não faz backup).
    """
    cfg = carregar_config()
    auto_promover = cfg.get("promover_modelos_apos_retreino", True)  # default True
    if not auto_promover:
        log_event("[RETREINO] Auto-promoção desativada por config (promover_modelos_apos_retreino=False).", level="info")
        return

    avaliacao = _carregar_avaliacao()
    flag_aprovado = _flag_novo_modelo_aprovado()
    if not avaliacao and not flag_aprovado:
        log_event("[RETREINO] Nenhuma avaliação/flag de aprovação encontrada. Não haverá promoção/backup.", level="info")
        return

    # caminhos canônicos (atuais)
    rf_canon = os.path.join(PROJETO_ROOT, "modelos", "cerebro_mestre.joblib")
    xgb_canon = os.path.join(PROJETO_ROOT, "modelos", "xgb_cerebro.json")
    lstm_canon = os.path.join(PROJETO_ROOT, "modelos", "lstm_cerebro.h5")

    # candidatos (timestampados)
    rf_cand = _listar_mais_recente(os.path.join("modelos", "cerebro_mestre_*.joblib"))
    xgb_cand = _listar_mais_recente(os.path.join("modelos", "xgb_cerebro_*.json"))
    lstm_cand = _listar_mais_recente(os.path.join("modelos", "lstm_cerebro_*.h5"))

    def _deve_promover(chave: str) -> bool:
        try:
            sub = avaliacao.get(chave, {})
            val = sub.get("promover", None)
            if val is None:
                return flag_aprovado  # fallback global
            return bool(val)
        except Exception:
            return flag_aprovado

    if rf_cand and os.path.exists(rf_cand) and _deve_promover("random_forest"):
        try:
            _promover(rf_canon, rf_cand)
        except Exception as e:
            log_event(f"[RETREINO] Falha ao promover RF: {e}", level="error")

    if xgb_cand and os.path.exists(xgb_cand) and _deve_promover("xgboost"):
        try:
            _promover(xgb_canon, xgb_cand)
        except Exception as e:
            log_event(f"[RETREINO] Falha ao promover XGB: {e}", level="error")

    if lstm_cand and os.path.exists(lstm_cand) and _deve_promover("lstm"):
        try:
            _promover(lstm_canon, lstm_cand)
        except Exception as e:
            log_event(f"[RETREINO] Falha ao promover LSTM: {e}", level="error")

    flag_path = os.path.join(PROJETO_ROOT, "modelos", "novo_modelo_aprovado.flag")
    if os.path.exists(flag_path):
        try:
            os.remove(flag_path)
            log_event("[RETREINO] Flag de aprovação removida após promoção.", level="info")
        except Exception as e:
            log_event(f"[RETREINO] Não foi possível remover flag de aprovação: {e}", level="warning")

# ========= (Fase 9) Gate Walk-Forward: helpers =========

def _carregar_walkforward_veredito() -> dict:
    """
    Procura veredito do WF em:
      - logs/walkforward_veredito.json
      - logs/wf_veredito.json
    Se não encontrar, tenta usar logs/walkforward_summary.json como fallback (parcial).
    """
    candidatos = [
        os.path.join(LOG_DIR, "walkforward_veredito.json"),
        os.path.join(LOG_DIR, "wf_veredito.json"),
    ]
    for caminho in candidatos:
        if os.path.exists(caminho):
            try:
                with open(caminho, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
            except Exception as e:
                log_event(f"[WF] Erro lendo {caminho}: {e}", level="warning")

    # Fallback parcial (não tem pf/winrate por fold, mas ainda dá pra auditar)
    summary = os.path.join(LOG_DIR, "walkforward_summary.json")
    if os.path.exists(summary):
        try:
            with open(summary, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            return {
                "perfil_vencedor": data.get("perfil_vencedor"),
                "f1_updown": data.get("f1_updown"),
                "neutral_rate": data.get("neutral_rate"),
                # sem pf_por_fold / acerto_por_fold
            }
        except Exception as e:
            log_event(f"[WF] Erro lendo summary fallback: {e}", level="warning")

    return {}

def _registrar_auditoria_promocao(payload: dict) -> str:
    """
    Salva auditoria do gate em logs/auditoria_promocao_YYYYmmdd_HHMMSS.json
    """
    destino = os.path.join(LOG_DIR, f"auditoria_promocao_{_ts()}.json")
    try:
        with open(destino, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_event(f"[RETREINO] Falha ao escrever auditoria: {e}", level="warning")
    return destino

# =============================================================================

def main():
    print("===> Iniciando pipeline de retreinamento do Robô IA FTMO...")
    log_event("==== Início do Pipeline de Retreino IA ====", level="info")
    enviar_alerta_telegram("🚀 Iniciando pipeline de retreinamento do Robô IA FTMO")

    etapas = [
        ("features/gerar_features.py", "Geração dos features para IA", True),
        ("mt5/coletar_candles_mt5.py", "Coleta/atualização dos candles", True),
        ("gestao/limpar_tickets_invalidos.py", "Limpeza de tickets inválidos", True),
        # ("gestao/fechar_todas_ordens.py", "Fechamento de ordens antigas (opcional)", False),
        ("modelos/treinar_modelos_ensemble.py", "Retreino dos modelos ENSEMBLE (RF/XGB/LSTM)", True),
        ("utils/diagnostico_sinais.py", "Diagnóstico dos sinais/resultados (opcional)", False),
        ("utils/auditor_features.py", "Auditoria dos features (opcional)", False)
    ]

    sucesso_geral = True
    for path, descricao, critico in etapas:
        if not rodar_script(path, descricao):
            sucesso_geral = False
            if critico:
                log_event(f"Falha crítica na etapa '{descricao}'. Pipeline interrompido.", level="error")
                break

    if sucesso_geral:
        # ===== promoção/backup apenas quando houver aprovação =====
        try:
            promover_modelos_se_aprovados()
        except Exception as e:
            log_event(f"[RETREINO] Erro ao executar promoção de modelos: {e}", level="error")

        # ===== (Fase 9) Gate Walk-Forward A/B + relatório =====
        try:
            ROOT = Path(PROJETO_ROOT)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT)  # garante import de utils/ e scripts/

            cfg = carregar_config()
            features_csv = ROOT / "dados" / "features.csv"
            logs_dir = ROOT / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            val_size = str(cfg.get("wf_val_size", 3000))
            anchored_flag = str(cfg.get("wf_anchored", True)).lower()  # "true"/"false"

            # 1) Walk-Forward + A/B -> cria logs/A, logs/B e promove vencedor para logs/
            #    (tenta como módulo e cai para script se necessário)
            cmds = [
                [PYTHON_EXEC, "-m", "scripts.run_walkforward_ab",
                 "--features", str(features_csv),
                 "--val_size", val_size,
                 "--anchored", anchored_flag,
                 "--out", str(logs_dir)],
                [PYTHON_EXEC, str(ROOT / "run_walkforward_ab.py"),
                 "--features", str(features_csv),
                 "--val_size", val_size,
                 "--anchored", anchored_flag,
                 "--out", str(logs_dir)],
            ]
            ok = False
            for cmd in cmds:
                try:
                    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
                    ok = True
                    break
                except subprocess.CalledProcessError as e:
                    log_event(f"[RETREINO] WF A/B falhou (tentativa {cmd}): {e}", level="warning")
            if not ok:
                raise RuntimeError("Walk-Forward A/B não executou com sucesso.")

            # 2) Relatório (só leitura; não crítico)
            try:
                subprocess.run(
                    [PYTHON_EXEC, "-m", "scripts.relatorio_walkforward", "--base", str(logs_dir)],
                    cwd=str(ROOT), env=env, check=False
                )
            except Exception as e:
                log_event(f"[PIPELINE][WF] Relatório de WF falhou (não crítico): {e}", level="warning")

            # 3) Gate: carrega veredito e aplica critérios
            ver = _carregar_walkforward_veredito()
            if not ver:
                log_event("[RETREINO] Veredito WF não encontrado; reprovando por segurança.", level="warning")
                aprovado = False
            else:
                # thresholds
                pf_min = float(cfg.get("wf_gate_pf_min", 1.2))
                acerto_min = float(cfg.get("wf_gate_acerto_min", 0.52))
                min_folds_ok = int(cfg.get("wf_gate_min_folds_ok", 2))

                try:
                    # usa a implementação oficial do gate (seu util)
                    from utils.promocao_modelo import gate_walkforward
                except Exception:
                    # fallback simples se util não estiver disponível
                    def gate_walkforward(v: dict, pf_min: float, acerto_min: float, min_folds_ok: int) -> bool:
                        pfs = v.get("pf_por_fold") or []
                        accs = v.get("acerto_por_fold") or []
                        ok = 0
                        for pf, ac in zip(pfs, accs):
                            if pf is None or ac is None:
                                continue
                            if pf >= pf_min and ac >= acerto_min:
                                ok += 1
                        return ok >= min_folds_ok

                aprovado = gate_walkforward(ver, pf_min, acerto_min, min_folds_ok)

            # 4) Auditoria + Telegram
            auditoria = {
                "ts": _ts(),
                "fase": "Fase 9 - Gate Walk-Forward",
                "veredito": ver or {},
                "aprovado": bool(aprovado),
                "criterios": {
                    "pf_min": float(cfg.get("wf_gate_pf_min", 1.2)),
                    "acerto_min": float(cfg.get("wf_gate_acerto_min", 0.52)),
                    "min_folds_ok": int(cfg.get("wf_gate_min_folds_ok", 2)),
                },
            }
            caminho_aud = _registrar_auditoria_promocao(auditoria)
            log_event(f"[RETREINO] Auditoria registrada em: {caminho_aud}", level="info")
            enviar_alerta_telegram(f"🧪 Gate Walk-Forward — {'APROVADO ✅' if aprovado else 'REPROVADO ❌'}\n"
                                   f"Auditoria: {os.path.basename(caminho_aud)}")

            if not aprovado:
                log_event("[RETREINO] Gate WF reprovado; promoção não será executada.", level="warning")
            else:
                log_event("[RETREINO] Gate WF aprovado; (promoção já é controlada pela etapa anterior com avaliação).", level="info")

        except Exception as e:
            log_event(f"[PIPELINE][WF] Falha no bloco de Walk-Forward (não crítico): {e}", level="warning")

        # ===== finalização =====
        print("===> Pipeline concluído com sucesso!")
        log_event("==== Pipeline finalizado com SUCESSO ====", level="info")
        enviar_alerta_telegram("🏁 Pipeline de retreinamento concluído com sucesso!")
    else:
        log_event("==== Pipeline finalizado COM ERROS. Verifique o log! ====", level="warning")
        enviar_alerta_telegram("⚠️ Pipeline de retreinamento finalizado com erros. Verifique os logs!")

if __name__ == "__main__":
    main()
