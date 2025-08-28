import time
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from colorama import Fore, Style, init

# === Importações do Projeto ===
from utils.utils import carregar_config, eh_dia_util, aguardar_inicio_novo_candle
from utils.eh_feriado import eh_feriado
from mt5.coletar_candles_mt5 import coletar_candles
from features.gerar_features import calcular_features
from inteligencia.estrategia_ia import gerar_sinal
from inteligencia.validar_tp_sl import validar_tp_sl_historico  # <-- (ajuste de import se necessário)
from utils.debug_logger import log_event, log_decisao  # <- AJUSTE: log_delay_execucao sai daqui
from utils.monitor_delay_execucao import log_delay_execucao   # <- NOVO: import correto

def _pode_abrir_nova_posicao(ativo: str, max_por_ativo: int) -> bool:
    try:
        poss = mt5.positions_get(symbol=ativo) or []
        return len(poss) < int(max_por_ativo)
    except Exception:
        return True

from utils.protecao_janela import check_e_acionar_protecao
from utils.status_snapshot import snapshot_status, formatar_status
from utils.mt5_watchdog import garantir_conexao_mt5 as watchdog_conexao
from utils.healthcheck import run_healthcheck_once
from utils.estado_execucao import tick_estado, tempo_restante_minutos, carregar_estado, deve_notificar_cooldown, registrar_notificacao_cooldown
from utils.lote_adaptativo import calcular_lote_adaptativo
from utils.vote_monitor import obter_multiplicador_lote_por_vies
from comunicacao.telegram_alertas import enviar_telegram
from comunicacao.telegram_bot import (
    checar_comando, ler_comando_telegram, setar_comando_telegram
)
from risco.risco_ftmo import verificar_risco
from inteligencia.ranking_padroes import atualizar_ranking, obter_score_padrao
from inteligencia.memoria_adaptativa import reforcar_memoria, obter_score_memoria
from inteligencia.clusterizar_regimes import detectar_regime
from inteligencia.contexto import analisar_contexto
from inteligencia.contexto_ordens import registrar_contexto_ordem
from gestao.gestao_posicoes import (
    obter_ordens_abertas_mt5, saldo_bruto, valor_investido, lucro_fechado,
    lucro_aberto, lucro_total, percentual_lucro,
)
from gestao.sincronizar_historico_completo import sincronizar_historico_completo
from gestao.sincronizar_historico_incremental import sincronizar_historico_incremental
from utils.monitor_inatividade import registrar_ciclo
from utils.backup_fallback import backup_modelo, backup_banco  # mantido (sem chamadas por ciclo)
from utils.sinal_utils import normalizar_sinal, sinal_to_str
from utils.operacao_institucional import abrir_ordem_e_registrar, fechar_ordem_e_registrar
from utils.db_schema_manager import garantir_schema_operacoes
from utils.protecao_loss import (
    atingiu_loss_diario, bloquear_entradas_loss, esta_bloqueado_loss,
)
from utils.protecao_loss_flutuante import (
    checar_loss_flutuante, bloquear_loss_flutuante, esta_bloqueado_loss_flutuante,
    modo_acao_loss_flutuante  # <- NOVO
)
from utils.horario_operacional import dentro_horario_operacao
from utils.meta_diaria import atingiu_meta_periodo
from utils.validador_robo import registrar_acao, validar_pendentes
from utils.tempo_ciclo import TempoCiclo

init(autoreset=True)

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

# --- helper robusto para converter timeframe textual em minutos ---
def _tf_to_minutes(tf: str) -> int:
    try:
        tf = (tf or "M1").upper().strip()
        if tf.endswith("MIN") or tf.endswith("M"):
            n = int("".join(ch for ch in tf if ch.isdigit()))
            return max(1, n)
        if tf.endswith("H") or tf.endswith("HOUR") or tf.endswith("HORAS") or tf.endswith("HORA"):
            n = int("".join(ch for ch in tf if ch.isdigit()))
            return max(1, n * 60)
        if tf.endswith("D") or tf.endswith("DAY") or tf.endswith("DIA") or tf.endswith("DIAS"):
            n = int("".join(ch for ch in tf if ch.isdigit()))
            return max(1, n * 60 * 24)
        # fallback: 1 minuto
        return 1
    except Exception:
        return 1


db_path = os.path.join("dados", "robodados.duckdb")
# ⬇️ Mudança: timestamp agora desejado como TIMESTAMPTZ (evita erro VARCHAR vs TIMESTAMPTZ)
colunas_necessarias = {
    "id": "BIGINT", "timestamp": "TIMESTAMPTZ", "ativo": "VARCHAR", "padrao": "VARCHAR",
    "regime": "VARCHAR", "contexto": "VARCHAR", "hora": "VARCHAR", "tipo": "VARCHAR",
    "volume": "FLOAT", "preco_abertura": "FLOAT", "preco_fechamento": "FLOAT",
    "score_reforco": "FLOAT", "motivo_saida": "VARCHAR", "sinal": "VARCHAR", "data_fechamento": "TIMESTAMP"
}
garantir_schema_operacoes(db_path, colunas_necessarias)

# Migração defensiva: se 'operacoes.timestamp' ainda for VARCHAR, tentar converter para TIMESTAMPTZ
def _migrar_timestamp_operacoes_para_timestamptz(caminho_db: str):
    try:
        import duckdb  # usado só aqui
        con = duckdb.connect(caminho_db)
        tipo = con.execute("""
            SELECT UPPER(data_type)
            FROM information_schema.columns
            WHERE table_name = 'operacoes' AND column_name = 'timestamp'
        """).fetchone()
        if tipo and "VARCHAR" in (tipo[0] or ""):
            # tenta converter automaticamente; valores inválidos viram NULL (TRY_CAST)
            con.execute("""
                ALTER TABLE operacoes
                ALTER COLUMN "timestamp" TYPE TIMESTAMPTZ
                USING TRY_CAST("timestamp" AS TIMESTAMPTZ)
            """)
            log_event("[DB] Migração: 'operacoes.timestamp' convertido para TIMESTAMPTZ.", level="info")
        con.close()
    except Exception as e:
        # não interrompe o robô por falha de migração
        log_event(f"[DB] Migração opcional do timestamp falhou (seguindo mesmo assim): {e}", level="warning")

_migrar_timestamp_operacoes_para_timestamptz(db_path)

# --- helper para extrair volume usado no envio ---
def _extrair_volume_usado(resultado, ativo):
    try:
        if isinstance(resultado, dict):
            diag = resultado.get("diagnostico", {}) or {}
            v = diag.get("volume_enviado")
            if v is not None:
                return float(v)
        # fallback ao config se o executor não reportou
        from utils.utils import carregar_config
        cfg = carregar_config()
        return float(cfg.get("volumes", {}).get(ativo, cfg.get("volume_padrao", 0.01)))
    except Exception:
        return None

# ==== Helper de rastreabilidade de bloqueios ====
def _log_bloqueio(ativo: str, ciclo: int, motivo: str, **extra):
    """
    Log padronizado de motivo de NÃO execução de ordem.
    Ex.: _log_bloqueio("EURUSD", 12, "risco_ftmo", detalhe="max loss diário")
    """
    base = f"[MOTIVO BLOQUEIO] ativo={ativo} ciclo={ciclo} motivo={motivo}"
    if extra:
        base += " | " + " | ".join(f"{k}={v}" for k, v in extra.items())
    log_event(base, level="warning")

# ==== Compat: tenta obter (score, N) do ranking. Se não houver função nova, não aplica limiar por N. ====
def _obter_score_e_n_compat(padrao: str, ativo: str):
    """
    Retorna (score_padrao, n_ocorrencias ou None).
    Tenta usar inteligencia.ranking_padroes.obter_score_e_n se existir.
    Caso contrário, usa obter_score_padrao e retorna n=None (não aplicar limiar por N).
    """
    score = 0.0
    n = None
    try:
        score = float(obter_score_padrao(padrao, ativo))
    except Exception:
        score = 0.0
    try:
        # import dinâmico para compatibilidade com versões antigas
        from inteligencia.ranking_padroes import obter_score_e_n  # type: ignore
        try:
            score2, n = obter_score_e_n(padrao, ativo)  # espera (float, int)
            score = float(score2)
        except Exception:
            pass
    except Exception:
        pass
    # clamp de segurança
    try:
        score = max(-1.0, min(1.0, float(score)))
    except Exception:
        score = 0.0
    return score, (int(n) if n is not None else None)


# === Helpers de STATUS/Telegram sempre-on (adicionados; sem dependências de variáveis locais) ===
def _snapshot_bloqueios_e_metricas():
    """Snapshot mínimo para o comando 'status' sem depender de variáveis locais."""
    try:
        flags = {
            "feriado": eh_feriado() if 'eh_feriado' in globals() else None,
            "loss_diario": esta_bloqueado_loss() if 'esta_bloqueado_loss' in globals() else None,
            "loss_flutuante": esta_bloqueado_loss_flutuante() if 'esta_bloqueado_loss_flutuante' in globals() else None,
            "meta_diaria": (atingiu_meta_periodo()[0] if 'atingiu_meta_periodo' in globals() else None),
            "fora_horario": None,
        }
        metrica = {"posicoes_totais": None, "posicoes_por_ativo": "", "saldo": None}
        try:
            if 'saldo_bruto' in globals():
                metrica["saldo"] = saldo_bruto()
        except Exception:
            pass
        return flags, metrica
    except Exception as e:
        return {"erro": str(e)}, {}

def processar_comandos_telegram_sempre():
    """Responde 'status'/'motivo'/'resumo' mesmo durante bloqueios/esperas."""
    try:
        checar_comando()
        cmd = ler_comando_telegram()
    except Exception:
        cmd = None
    if not cmd:
        return
    c = str(cmd).strip().lower()
    try:
        if c in ("status", "resumo", "motivo", "heartbeat"):
            snap = snapshot_status()
            txt = formatar_status(snap, compacto=(c=="heartbeat"))
            try:
                enviar_telegram(txt)
            except Exception:
                pass
    except Exception as e:
        try:
            enviar_telegram(f"[STATUS] Falha ao gerar resumo: {e}")
        except Exception:
            pass

def garantir_conexao_mt5(intervalo_reconexao=5):
    """Wrapper simples para manter compatibilidade com o watchdog, se existir."""
    try:
        # Prefira watchdog se disponível
        return watchdog_conexao(intervalo_reconexao=intervalo_reconexao)
    except Exception:
        # Fallback local
        if mt5.initialize():
            msg_ok = "✅ Reconexão com o MetaTrader 5 estabelecida! Retomando operação."
            print(Fore.GREEN + msg_ok + Style.RESET_ALL)
            log_event(msg_ok, level="info")
            return True
        else:
            while not mt5.initialize():
                msg = f"❌ Perda de conexão com o MetaTrader 5! Tentando reconectar em {intervalo_reconexao} min..."
                print(Fore.RED + msg + Style.RESET_ALL)
                log_event(msg, level="critical")
                try:
                    enviar_telegram(msg)
                except Exception:
                    pass
                time.sleep(intervalo_reconexao * 60)
            msg_ok = "✅ Reconexão com o MetaTrader 5 estabelecida! Retomando operação."
            print(Fore.GREEN + msg_ok + Style.RESET_ALL)
            log_event(msg_ok, level="info")
            try:
                enviar_telegram(msg_ok)
            except Exception:
                pass
            return True

def ja_sincronizou_hoje():
    controle_path = "dados/ultimo_sync_historico.txt"
    hoje = datetime.now().date()
    if os.path.exists(controle_path):
        with open(controle_path, "r") as f:
            ultima_data = f.read().strip()
            if ultima_data == str(hoje):
                return True
    return False

def marcar_sincronizado():
    controle_path = "dados/ultimo_sync_historico.txt"
    hoje = datetime.now().date()
    with open(controle_path, "w") as f:
        f.write(str(hoje))

def mercado_aberto():
    agora = datetime.now()
    if agora.weekday() == 4 and agora.hour >= 18:
        return False
    if agora.weekday() == 5:
        return False
    if agora.weekday() == 6 and agora.hour < 19:
        return False
    return True

def fechar_todas_ordens(ativos, motivo="Fechamento automático por proteção de risco"):
    total_fechadas = 0
    for ativo in ativos:
        ordens_abertas = obter_ordens_abertas_mt5(ativo)
        for ordem in ordens_abertas:
            ticket = ordem.get('ticket', ordem.get('order', ordem.get('ticket')))
            preco_atual = ordem.get('price_open')
            resultado_fechamento = fechar_ordem_e_registrar(
                ticket=ticket,
                ativo=ativo,
                preco_fechamento=preco_atual,
                motivo_fechamento=motivo
            )
            log_event(f"Ordem {ticket} de {ativo} FECHADA por proteção. Resultado: {resultado_fechamento}", level="warning")
            try:
                atualizar_ranking(ativo, resultado_fechamento)
                log_event(f"[RANKING] Ranking atualizado após fechamento da ordem {ticket} no ativo {ativo}", level="info")
            except Exception as e:
                log_event(f"[RANKING] Erro ao atualizar ranking após fechamento da ordem {ticket}: {e}", level="error")
            total_fechadas += 1
    return total_fechadas

def sanity_check_dict(dados, colunas_check=None):
    if colunas_check is None:
        colunas_check = dados.keys()
    erros = []
    for k in colunas_check:
        v = dados.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == ""):
            erros.append(k)
    return erros

def gravar_ultimo_ciclo_log(info_dict):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    df = pd.DataFrame([info_dict])
    df.to_csv(os.path.join(log_dir, "ultimo_ciclo.csv"), index=False, sep=";")

def get_pip_factor(config, ativo):
    pip_factors = config.get("pip_factors", {})
    if isinstance(pip_factors, dict):
        return pip_factors.get(ativo, 0.0001)
    return 0.0001

def main():
    config = carregar_config()
    ativos = config.get("ativos", ["EURUSD"])
    timeframes = config.get("timeframes", {})
    janela_candles = config.get("janela_candles", 96)
    caminho_modelo = "modelos/cerebro_mestre.joblib"
    caminho_banco = "dados/robodados.duckdb"
    saldo_inicial = config.get("capital_conta", 10000)
    limite_loss_dia = -abs(saldo_inicial * (config.get("limite_loss_dia_percentual", 2.5) / 100))
    # <- AJUSTE: lê novo e legado
    max_ordens_loss_aberto = (config.get("protecao_flutuante", {}) or {}).get("max_ordens_loss_aberto",
                               config.get("max_ordens_loss_flutuante", 3))
    horarios_operacao = config.get("horarios_operacao", [{"inicio": "00:00", "fim": "23:59"}])
    meta_gain = config.get("meta_gain_dia", 2)
    meta_loss = config.get("meta_loss_dia", -2)
    pausar_feriado = config.get("pausar_feriado", True)
    intervalo_reconexao_mt5 = config.get("intervalo_reconexao_mt5_min", 5)
    intervalo_checagem_mercado_fechado = config.get("intervalo_checagem_mercado_fechado_min", 20)

    # === Novos/ajustados parâmetros ===
    LIMIAR_SCORE_PADRAO = config.get("limiar_score_padrao", 0.2)
    PESO_SCORE_PADRAO = config.get("peso_score_padrao", 0.7)
    PESO_SCORE_MEMORIA = config.get("peso_score_memoria", 0.3)
    MIN_EVID_PADRAO = int(config.get("min_evid_padroes", 20))  # só aplica limiar quando N >= este valor
    TP_SL_BLOCK_CONF = float(config.get("tp_sl_block_confidence", 0.6))  # confiança mínima para bloquear por TP/SL preview
    TP_SL_PENALTY = float(config.get("tp_sl_penalty", 0.10))  # penalidade "advisory" no score_total

    # === Cooldown inicial (minutos) ===
    filtros_cfg = (config.get("filtros", {}) if isinstance(config, dict) else {}) or {}
    cooldown_minutos = int(filtros_cfg.get("cooldown_minutos", config.get("cooldown_minutos", 0)))
    inicio_execucao = datetime.now()

    print(Fore.YELLOW + "🚀 Robô FTMO INSTITUCIONAL iniciado." + Style.RESET_ALL)
    log_event("Robô iniciado", level="info")
    garantir_conexao_mt5(intervalo_reconexao_mt5)

    try:
        enviar_telegram("🚀 Robô FTMO INSTITUCIONAL iniciado.")
    except Exception as e:
        log_event(f"Falha ao enviar mensagem de início ao Telegram: {e}", level="error")

    if not os.path.exists(caminho_modelo):
        alerta = "❌ Modelo de IA (cerebro_mestre.joblib) não encontrado! Treine antes de rodar!"
        print(Fore.RED + alerta + Style.RESET_ALL)
        log_event(alerta, level="error")
        try:
            enviar_telegram(alerta)
        except Exception as e:
            log_event(f"Falha ao enviar alerta de falta de modelo ao Telegram: {e}", level="error")
        return

    ciclo_num = 0
    fechamento_sexta_realizado = False
    fora_horario_notificado = False

    try:
        while True:

            tc = TempoCiclo()
            tc.iniciar_ciclo()
            ciclo_t0 = time.perf_counter()  # início do ciclo (monotônico)
            operou = False

            # === Telegram: responde comandos SEMPRE, início do ciclo ===
            try:
                processar_comandos_telegram_sempre()
            except Exception:
                pass
            bloqueios_status = {}

            timeframe_min = min([_tf_to_minutes(timeframes.get(a, 'M1')) for a in ativos])
            tc.iniciar_espera()
            aguardar_inicio_novo_candle(timeframe_min)
            tc.finalizar_espera()

            garantir_conexao_mt5(intervalo_reconexao_mt5)

            # Estado de cooldown deste ciclo
            em_cooldown = False
            if cooldown_minutos > 0:
                try:
                    em_cooldown = (datetime.now() - inicio_execucao) < timedelta(minutes=cooldown_minutos)
                    if em_cooldown:
                        try:
                            if deve_notificar_cooldown():
                                enviar_telegram(f"⏳ Em cooldown inicial de {cooldown_minutos} min. Sem novas entradas.")
                                registrar_notificacao_cooldown()
                        except Exception:
                            pass
                except Exception:
                    em_cooldown = False

            # BLOQUEIOS INSTITUCIONAIS
            bloqueios_status['feriado'] = eh_feriado()
            bloqueios_status['loss_diario'] = esta_bloqueado_loss()
            # Modo de ação (soft/hard) para flutuante
            modo_flut = "hard"
            try:
                modo_flut = modo_acao_loss_flutuante()
            except Exception:
                modo_flut = "hard"
            bloqueios_status['loss_flutuante'] = esta_bloqueado_loss_flutuante()
            atingiu_meta_flag, tipo_meta, _ = atingiu_meta_periodo()
            bloqueios_status['meta_diaria'] = atingiu_meta_flag
            bloqueios_status['fora_horario'] = not dentro_horario_operacao(horarios_operacao)

            # ⬇️ Hotfix: agregador defensivo para risco FTMO (não trava se houver erro na query DuckDB)
            try:
                risco_ok_agg = True
                for _a in ativos:
                    try:
                        if not verificar_risco(_a):
                            risco_ok_agg = False
                            break
                    except Exception as e:
                        log_event(f"[RISCO] Erro ao verificar risco FTMO (agregador) para {_a}: {e}", level="error")
                        continue
                bloqueios_status['risco_ftmo'] = not risco_ok_agg
            except Exception as e:
                log_event(f"[RISCO] Falha no agregador de risco FTMO: {e}", level="error")
                bloqueios_status['risco_ftmo'] = False

            # Log técnico bruto (JSON)
            import json
            os.makedirs("logs", exist_ok=True)
            with open("logs/bloqueios_status.json", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": str(datetime.now()),
                    **bloqueios_status
                }) + "\n")
            log_event(f"[TRAVAS] Status do ciclo: {bloqueios_status}", level="debug")

            # --- FERIADO ---
            if pausar_feriado and eh_feriado():
                _log_bloqueio("-", ciclo_num, "feriado")
                msg = "Robô pausado por feriado. Nenhuma operação será realizada hoje."
                log_event(msg, level="warning")
                try:
                    enviar_telegram("⏸️ " + msg)
                except Exception:
                    pass
                agora = datetime.now()
                amanha = (agora + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
                time.sleep((amanha - agora).total_seconds())
                continue

            # --- FECHAMENTO DE SEXTA ---
            agora = datetime.now()
            if agora.weekday() == 4:
                if (agora.hour == 17 and agora.minute >= 50) or (agora.hour == 18 and agora.minute == 0):
                    if not fechamento_sexta_realizado:
                        total_fechadas = fechar_todas_ordens(
                            ativos,
                            motivo="Fechamento obrigatório sexta-feira (10 minutos antes do mercado fechar)"
                        )
                        if total_fechadas > 0:
                            operou = True
                            msg = (
                                f"🔔 Todas as ordens ({total_fechadas}) foram fechadas por rotina obrigatória de sexta-feira, "
                                f"10 minutos antes do mercado encerrar."
                            )
                            log_event(msg, level="warning")
                            try:
                                enviar_telegram(msg)
                            except Exception:
                                pass
                        fechamento_sexta_realizado = True
                else:
                    fechamento_sexta_realizado = False
            else:
                fechamento_sexta_realizado = False

            # --- MERCADO FECHADO ---
            if not mercado_aberto():
                _log_bloqueio("-", ciclo_num, "mercado_fechado", proxima_checagem_min=intervalo_checagem_mercado_fechado)
                log_event(
                    f"⏸️ Mercado fechado! Hibernando {intervalo_checagem_mercado_fechado} minutos até a próxima checagem de abertura.",
                    level="info"
                )
                try:
                    enviar_telegram(
                        f"⏸️ Mercado fechado. Próxima checagem de abertura em {intervalo_checagem_mercado_fechado} min."
                    )
                except Exception:
                    pass
                time.sleep(intervalo_checagem_mercado_fechado * 60)
                continue

            # --- BLOQUEIO LOSS DIÁRIO (flag persistente) ---
            if esta_bloqueado_loss():
                _log_bloqueio("-", ciclo_num, "bloqueio_loss_diario")
                log_event("Robô bloqueado por proteção de loss diário. Aguardando virar o dia para retomar.", level="warning")
                try:
                    enviar_telegram("⚠️ Robô em modo de proteção: atingido o limite de loss diário. Só irá operar novamente amanhã.")
                except Exception:
                    pass
                while esta_bloqueado_loss():
                    time.sleep(120)
                log_event("Proteção de loss diário liberada, retomando operação.", level="info")
                try:
                    enviar_telegram("Robô liberado após proteção de loss diário. Retomando operação.")
                except Exception:
                    pass
                continue

            # --- HARD STOP por perda do dia ---
            atingiu, perda_dia = atingiu_loss_diario(limite_loss_dia)
            if atingiu:
                _log_bloqueio("-", ciclo_num, "hard_loss_diario", perda_dia=f"{perda_dia:.2f}", limite=f"{limite_loss_dia:.2f}")
                log_event(f"Proteção HARD ativada: perda do dia = {perda_dia:.2f} <= limite {limite_loss_dia:.2f}. Fechando tudo e hibernando...", level="critical")
                try:
                    enviar_telegram(f"🚨 Proteção HARD: Perda do dia = {perda_dia:.2f} <= limite {limite_loss_dia:.2f}. Todas as ordens serão fechadas e o robô será hibernado até amanhã.")
                except Exception:
                    pass
                total_fechadas = fechar_todas_ordens(ativos, motivo="Fechamento automático por atingimento do loss diário")
                bloquear_entradas_loss()
                log_event(f"Total de ordens fechadas por proteção: {total_fechadas}", level="critical")
                if total_fechadas > 0:
                    operou = True
                continue

            # ======= LOSS FLUTUANTE =======
            bloqueio_novas_entradas = False  # <- chave para SOFT
            # --- BLOQUEIO FLUTUANTE (flag persistente) ---
            if esta_bloqueado_loss_flutuante():
                if modo_flut == "hard":
                    _log_bloqueio("-", ciclo_num, "bloqueio_loss_flutuante(HARD)")
                    log_event("Robô bloqueado por proteção de loss flutuante (HARD). Esperando virar o dia.", level="warning")
                    try:
                        enviar_telegram("⚠️ Robô em proteção (HARD): limite de ordens em prejuízo simultâneo. Aguardando amanhã.")
                    except Exception:
                        pass
                    while esta_bloqueado_loss_flutuante():
                        time.sleep(120)
                    log_event("Proteção de loss flutuante liberada, retomando operação.", level="info")
                    try:
                        enviar_telegram("Robô liberado após proteção de loss flutuante. Retomando operação.")
                    except Exception:
                        pass
                    continue
                else:
                    # SOFT: segue rodando, mas sem abrir novas entradas
                    bloqueio_novas_entradas = True
                    _log_bloqueio("-", ciclo_num, "bloqueio_loss_flutuante(SOFT)")
                    log_event("Proteção SOFT ativa: novas ENTRADAS bloqueadas; gestão de posições continua.", level="warning")

            # --- DISPARO FLUTUANTE (atingido agora) ---
            atingiu_flutuante, qtd_loss, tickets_loss = checar_loss_flutuante(ativos, max_ordens_loss_aberto)
            if atingiu_flutuante:
                if modo_flut == "hard":
                    _log_bloqueio("-", ciclo_num, "hard_loss_flutuante", qtd_loss=qtd_loss, tickets=tickets_loss)
                    log_event(f"PROTEÇÃO HARD: {qtd_loss} ordens negativas simultâneas! Fechando todas e bloqueando o dia.", level="critical")
                    try:
                        enviar_telegram(f"🚨 PROTEÇÃO FLUTUANTE (HARD): {qtd_loss} ordens em prejuízo! Fechando todas e hibernando até amanhã.\nTickets: {tickets_loss}")
                    except Exception:
                        pass
                    total_fechadas = fechar_todas_ordens(ativos, motivo="Fechamento automático por proteção de loss flutuante (HARD)")
                    bloquear_loss_flutuante()
                    log_event(f"Total de ordens fechadas por proteção flutuante (HARD): {total_fechadas}", level="critical")
                    if total_fechadas > 0:
                        operou = True
                    continue
                else:
                    # SOFT: não fecha, apenas bloqueia novas entradas
                    _log_bloqueio("-", ciclo_num, "soft_loss_flutuante", qtd_loss=qtd_loss, tickets=tickets_loss)
                    log_event(f"PROTEÇÃO SOFT: {qtd_loss} ordens negativas simultâneas. Bloqueando NOVAS entradas (ordens abertas permanecem).", level="warning")
                    try:
                        enviar_telegram(f"⚠️ PROTEÇÃO FLUTUANTE (SOFT): {qtd_loss} ordens em prejuízo — bloqueando NOVAS entradas. Tickets: {tickets_loss}")
                    except Exception:
                        pass
                    bloquear_loss_flutuante()  # marca o dia como bloqueado (para entradas)
                    bloqueio_novas_entradas = True
                    # segue sem 'continue': gestão de posições permanece ativa

            # --- META DO DIA (gain/loss) ---
            atingiu_meta, tipo_meta, pct = atingiu_meta_periodo()
            if atingiu_meta:
                _log_bloqueio("-", ciclo_num, f"meta_{tipo_meta}_atingida", percentual=f"{pct:.2f}%")
                if tipo_meta == "gain":
                    log_event(f"Meta de ganho diário atingida ({pct:.2f}%). Pausando operações até amanhã.", level="critical")
                    try:
                        enviar_telegram(f"🏆 Meta de GAIN diário atingida ({pct:.2f}%). Robô pausado até amanhã!")
                    except Exception:
                        pass
                elif tipo_meta == "loss":
                    log_event(f"Meta de loss diário atingida ({pct:.2f}%). Pausando operações até amanhã.", level="critical")
                    try:
                        enviar_telegram(f"🚨 Meta de LOSS diário atingida ({pct:.2f}%). Robô pausado até amanhã!")
                    except Exception:
                        pass
                while True:
                    agora = datetime.now()
                    if agora.hour == 0:
                        break
                    time.sleep(60 * 10)
                continue

            # --- HORÁRIO OPERACIONAL ---
            if not dentro_horario_operacao(horarios_operacao):
                _log_bloqueio("-", ciclo_num, "fora_horario_operacional")
                if not fora_horario_notificado:
                    log_event("Fora do horário operacional. Aguardando janela de operação...", level="info")
                    try:
                        enviar_telegram("⏸️ Fora do horário operacional. Robô aguardando janela permitida para operar.")
                    except Exception:
                        pass
                    fora_horario_notificado = True
                time.sleep(60)
                continue
            else:
                if fora_horario_notificado:
                    log_event("Dentro do horário operacional. Robô retomando operações.", level="info")
                    try:
                        enviar_telegram("▶️ Dentro do horário operacional. Robô retomando operações.")
                    except Exception:
                        pass
                    fora_horario_notificado = False

            # --- SYNC de madrugada ---
            agora = datetime.now()
            if agora.hour == 2 and not ja_sincronizou_hoje():
                try:
                    sincronizar_historico_completo()
                    marcar_sincronizado()
                    log_event(f"[SYNC HISTORICO] Sincronismo total do histórico MT5 executado às {agora.strftime('%Y-%m-%d %H:%M:%S')}")
                except Exception as e:
                    log_event(f"[SYNC HISTORICO] ERRO no sincronismo total: {e}")

            # === INÍCIO DO CICLO DE DECISÃO ===
            ciclo_num += 1
            ciclo_info_log = {
                "timestamp": datetime.now().isoformat(),
                "ciclo_num": ciclo_num
            }
            try:
                registrar_ciclo()

                # (REMOVIDO) backup por ciclo — não é mais executado aqui

                try:
                    log_event(f"Iniciando sincronização incremental de ordens (CICLO {ciclo_num})", level="info")
                    sincronizar_historico_incremental(minutos=60)
                except Exception as e:
                    log_event(f"Erro ao sincronizar ordens (CICLO {ciclo_num}): {e}", level="error")

                if not eh_dia_util():
                    log_event(f"Fora de dia útil (CICLO {ciclo_num}), aguardando...", level="info")
                    time.sleep(60)
                    continue

                log_event(f"Ciclo {ciclo_num} - início", level="info")

                for ativo in ativos:
                    timeframe_str = timeframes.get(ativo, config.get("timeframe", "M1"))
                    timeframe = TIMEFRAME_MAP.get(timeframe_str, mt5.TIMEFRAME_M1)
                    candles = coletar_candles(ativo, quantidade=janela_candles, timeframe=timeframe)
                    if candles is None or candles.empty:
                        log_event(f"Erro: candles vazios para {ativo} (tf={timeframe_str}) (CICLO {ciclo_num})", level="warning")
                        ciclo_info_log[f"{ativo}_status"] = "candles_vazios"
                        continue
                    candles_feat = calcular_features(candles, config, ativo=ativo)
                    if candles_feat is None or candles_feat.empty:
                        log_event(f"Erro ao gerar features para {ativo} (CICLO {ciclo_num})", level="warning")
                        ciclo_info_log[f"{ativo}_status"] = "features_vazias"
                        continue

                    # --- Validar pendentes N+1 (do ciclo anterior) para este ativo ---
                    try:
                        df_candles_val = candles.copy()
                        if "datahora" not in df_candles_val.columns:
                            if "timestamp" in df_candles_val.columns:
                                df_candles_val["datahora"] = pd.to_datetime(df_candles_val["timestamp"])
                            elif "time" in df_candles_val.columns:
                                df_candles_val["datahora"] = pd.to_datetime(df_candles_val["time"], unit="s", errors="coerce")
                        n_validadas = validar_pendentes(df_candles_val, ativo)
                        if n_validadas > 0:
                            log_event(f"[VALIDAR] {ativo}: {n_validadas} linha(s) N+1 validadas neste ciclo")
                    except Exception as e:
                        log_event(f"[VALIDADOR] Erro ao validar N+1 para {ativo}: {e}", level="error")

                    # --- Regime ---
                    try:
                        regime = detectar_regime(candles_feat)
                        log_event(f"Regime de mercado detectado para {ativo}: {regime} (CICLO {ciclo_num})", level="info")
                    except Exception as e:
                        regime = "indefinido"
                        log_event(f"Erro ao detectar regime para {ativo}: {e} (CICLO {ciclo_num})", level="error")

                    # --- Risco FTMO por ativo (já defensivo) ---
                    try:
                        if not verificar_risco(ativo):
                            _log_bloqueio(ativo, ciclo_num, "risco_ftmo")
                            alerta_risco = f"🚨 [Risco FTMO] Operação bloqueada em {ativo}! Limite atingido."
                            log_event(alerta_risco + f" (CICLO {ciclo_num})", level="warning")
                            ciclo_info_log[f"{ativo}_status"] = "bloqueado_risco"
                            continue
                    except Exception as e:
                        log_event(f"Erro ao verificar risco para {ativo}: {e} (CICLO {ciclo_num})", level="error")

                    # --- Contexto ---
                    contexto = None
                    try:
                        contexto = analisar_contexto(candles_feat)
                        if isinstance(contexto, dict):
                            contexto['regime'] = regime
                        log_event(f"Contexto previsto para {ativo}: {contexto} (CICLO {ciclo_num})", level="info")
                    except Exception as e:
                        log_event(f"Sem previsão de contexto para {ativo}: {e} (CICLO {ciclo_num})", level="warning")

                    # --- Gestão de posições abertas (exemplo simples) ---
                    ordens_abertas = obter_ordens_abertas_mt5(ativo)
                    saldo_b = saldo_bruto()
                    investimento = valor_investido(ativo)
                    lucro_realizado = lucro_fechado(ativo)
                    pnl_aberto = lucro_aberto(ativo)
                    lucro_geral = lucro_total(ativo)
                    percentual = percentual_lucro(saldo_inicial, ativo)

                    for ordem in ordens_abertas:
                        try:
                            ticket = ordem.get('ticket', ordem.get('order', ordem.get('ticket')))
                            preco_abertura = ordem.get('price_open')
                            tipo_ordem = ordem.get('type')
                            preco_atual = candles_feat.iloc[-1]["close"] if not candles_feat.empty else None
                            fechar = False
                            motivo_fechamento = ""
                            if tipo_ordem == 0 and preco_atual >= preco_abertura * 1.001:
                                fechar = True
                                motivo_fechamento = "Lucro alvo 0.1% atingido"
                            if tipo_ordem == 1 and preco_atual <= preco_abertura * 0.999:
                                fechar = True
                                motivo_fechamento = "Lucro alvo 0.1% atingido"
                            if fechar:
                                resultado_fechamento = fechar_ordem_e_registrar(
                                    ticket=ticket,
                                    ativo=ativo,
                                    preco_fechamento=preco_atual,
                                    motivo_fechamento=motivo_fechamento
                                )
                                log_event(f"Ordem {ticket} fechada: {motivo_fechamento} - resultado: {resultado_fechamento}", level="info")
                                operou = True
                                try:
                                    atualizar_ranking(ativo, resultado_fechamento)
                                    log_event(f"[RANKING] Ranking atualizado após fechamento da ordem {ticket} no ativo {ativo}", level="info")
                                except Exception as e:
                                    log_event(f"Erro ao atualizar ranking após fechar ordem {ticket}: {e}", level="error")
                        except Exception as e:
                            log_event(f"Erro ao tentar fechar ordem {ordem.get('ticket')}: {e}", level="error")

                    # --- Geração de sinal (IA / ensemble) ---
                    resultado_sinal = gerar_sinal(candles_feat, ativo, contexto=contexto)
                    ciclo_info_log[f"{ativo}_input"] = {
                        "ordens_abertas": len(ordens_abertas),
                        "saldo_bruto": saldo_b,
                        "investimento": investimento,
                        "lucro_realizado": lucro_realizado,
                        "pnl_aberto": pnl_aberto,
                        "lucro_total": lucro_geral,
                        "percentual": percentual,
                        "features_ult_candle": candles_feat.iloc[-1].to_dict() if not candles_feat.empty else {},
                        "contexto": contexto
                    }

                    # --- Registrar ação do ciclo t (para validar em t+1) ---
                    try:
                        features_dict = candles_feat.iloc[-1].to_dict()
                        registrar_acao(
                            datahora=features_dict.get("timestamp", datetime.now().isoformat()),
                            ativo=ativo,
                            sinal_robo=resultado_sinal.get("sinal", 0) if resultado_sinal else 0,
                            preco_entrada=features_dict.get("close", None),
                            padrao=resultado_sinal.get("padrao", "") if resultado_sinal else "",
                            confianca=resultado_sinal.get("confianca", None) if resultado_sinal else None,
                            motivo=resultado_sinal.get("motivo", "") if resultado_sinal else ""
                        )
                    except Exception as e:
                        log_event(f"[VALIDADOR] Falha ao registrar ação do ciclo t para {ativo}: {e}", level="error")

                    # --- Checagem neutro ---
                    if (resultado_sinal is None) or (normalizar_sinal(resultado_sinal.get("sinal")) == 0):
                        _log_bloqueio(ativo, ciclo_num, "sinal_neutro")
                        log_event(f"Nenhum sinal válido (neutro) para {ativo} (CICLO {ciclo_num})", level="warning")
                        ciclo_info_log[f"{ativo}_sinal"] = "neutro"
                        continue

                    # === Gates de entrada: cooldown e proteção SOFT ===
                    if em_cooldown:
                        _log_bloqueio(ativo, ciclo_num, "cooldown_inicial", minutos=cooldown_minutos)
                        log_event(f"[COOLDOWN] Sem novas entradas para {ativo} por {cooldown_minutos} minutos após o start.", level="info")
                        continue

                    if bloqueio_novas_entradas:
                        _log_bloqueio(ativo, ciclo_num, "loss_flutuante_soft")
                        log_event(f"[SOFT] Entrada bloqueada em {ativo} pelo modo SOFT de loss flutuante. Gestão continua.", level="info")
                        continue

                    try:
                        sinal_int = normalizar_sinal(resultado_sinal["sinal"])
                        sinal_traduzido = sinal_to_str(sinal_int)
                        padrao = resultado_sinal.get("padrao", "")

                        # === Score de padrão + N de evidências (compatível com versões antigas) ===
                        score_padrao, n_ocorrencias = _obter_score_e_n_compat(padrao, ativo)

                        # Memória adaptativa
                        score_memoria = obter_score_memoria(ativo, padrao)
                        try:
                            score_memoria = float(score_memoria)
                            score_memoria = max(-1.0, min(1.0, score_memoria))
                        except Exception:
                            score_memoria = 0.0

                        # Ponderação padrao/memória
                        peso_padrao = PESO_SCORE_PADRAO
                        peso_memoria = PESO_SCORE_MEMORIA
                        total_pesos = peso_padrao + peso_memoria
                        if total_pesos != 1.0 and total_pesos > 0:
                            peso_padrao /= total_pesos
                            peso_memoria /= total_pesos

                        score_total = peso_padrao * score_padrao + peso_memoria * score_memoria

                        # --- Preview TP/SL (forward) — "advisory"
                        tp_pips = config.get("tp_pips", 10)
                        sl_pips = config.get("sl_pips", 10)
                        pip_factor = get_pip_factor(config, ativo)

                        prioridade_intracandle = str(
                            (resultado_sinal or {}).get("tp_sl_priority",
                                config.get("tp_sl_intracandle_priority", "SL"))
                        ).upper()

                        candles_futuros = coletar_candles(ativo, quantidade=10, timeframe=timeframe)
                        if candles_futuros is None or candles_futuros.empty:
                            resultado_tp_sl = "none"
                        else:
                            resultado_tp_sl = validar_tp_sl_historico(
                                candles_futuros,
                                preco_entrada=candles_feat.iloc[-1]["close"],
                                tipo_ordem="compra" if sinal_int == 1 else "venda",
                                tp_pips=tp_pips,
                                sl_pips=sl_pips,
                                ponto_pip=pip_factor,
                                prioridade_intracandle=prioridade_intracandle
                            )

                        score_total_ajust = score_total
                        if resultado_tp_sl == "loss":
                            score_total_ajust = max(-1.0, score_total - TP_SL_PENALTY)

                        # --- FILTROS DE EXECUÇÃO ---
                        confianca_exec = float(resultado_sinal.get("confianca", 0.5))
                        aplica_limiar_por_n = (n_ocorrencias is not None and n_ocorrencias >= MIN_EVID_PADRAO)

                        block_por_score = aplica_limiar_por_n and (score_total_ajust < LIMIAR_SCORE_PADRAO)
                        block_por_tp_sl = aplica_limiar_por_n and (resultado_tp_sl == "loss") and (confianca_exec >= TP_SL_BLOCK_CONF)

                        if block_por_score or block_por_tp_sl:
                            motivos = []
                            if block_por_tp_sl:
                                motivos.append(f"forward_tp_sl=loss_conf≥{TP_SL_BLOCK_CONF}_prio={prioridade_intracandle}")
                            if block_por_score:
                                motivos.append(f"score_baixo={score_total_ajust:.2f}<{LIMIAR_SCORE_PADRAO:.2f} (N={n_ocorrencias})")

                            _log_bloqueio(
                                ativo, ciclo_num, "filtros_execucao",
                                motivo="+".join(motivos),
                                padrao=padrao,
                                n_ocorrencias=n_ocorrencias,
                                score_padrao=f"{score_padrao:.2f}",
                                score_memoria=f"{score_memoria:.2f}",
                                score_total=f"{score_total:.2f}",
                                score_total_ajust=f"{score_total_ajust:.2f}",
                                tp_sl_preview=resultado_tp_sl,
                                tp_sl_prioridade=prioridade_intracandle,
                                confianca=confianca_exec
                            )

                            log_event(
                                f"[FILTRO EXECUÇÃO] Bloqueada | ativo={ativo} | motivo={'+'.join(motivos)} | "
                                f"padrao={padrao} | N={n_ocorrencias} | score_padrao={score_padrao:.2f} | "
                                f"score_memoria={score_memoria:.2f} | score_total={score_total:.2f} | "
                                f"score_total_ajust={score_total_ajust:.2f} | tp_sl_preview={resultado_tp_sl} "
                                f"(prio={prioridade_intracandle}) | confianca={confianca_exec:.2f} | "
                                f"sinal={sinal_traduzido} | ciclo={ciclo_num}",
                                level="warning"
                            )

                            ciclo_info_log[f"{ativo}_sinal"] = {
                                "padrao": padrao,
                                "n_ocorrencias": n_ocorrencias,
                                "score_total": score_total,
                                "score_total_ajust": score_total_ajust,
                                "resultado_tp_sl": resultado_tp_sl,
                                "tp_sl_prioridade": prioridade_intracandle,
                                "sinal_aprovado": False
                            }
                            continue

                        # --- Cálculo de volume e envio de ordem ---
                        saldo_atual = saldo_b
                        volatilidade = candles_feat["close"].std() if not candles_feat.empty else 1.0

                        # Confianca para execução
                        confianca_exec = max(0.0, min(1.0, float(resultado_sinal.get("confianca", 0.5))))

                        # Limite de exposição por ativo
                        max_por_ativo = int(config.get('protecao', {}).get('max_posicoes_por_ativo', 1))
                        if not _pode_abrir_nova_posicao(ativo, max_por_ativo):
                            log_event(f"[EXPOSICAO] Limite atingido para {ativo}: max {max_por_ativo} posições. Sinal ignorado.", level="info")
                            try:
                                enviar_telegram(f"ℹ️ Limite de exposição: não abri nova posição em {ativo} (max={max_por_ativo}).")
                            except Exception:
                                pass
                            continue  # <-- importante: nada a fazer se não pode abrir

                        # --- Lote adaptativo + redução por viés (Fase 7) ---
                        try:
                            volume_base = calcular_lote_adaptativo(
                                ativo, saldo_atual, volatilidade, confianca_exec, config
                            )
                        except Exception:
                            volume_base = float(config.get('volumes', {}).get(ativo, config.get('volume_padrao', 0.01)))
                        try:
                            mult_vies = float(obter_multiplicador_lote_por_vies(ativo, config))
                        except Exception:
                            mult_vies = 1.0
                        volume_final = max(0.01, float(volume_base) * float(mult_vies))

                        # Contexto que segue no registro institucional
                        ctx_exec = {
                            "regime": regime,
                            "contexto": contexto,
                            "hora": time.strftime("%H:%M:%S"),
                            "confianca": confianca_exec,
                            "mult_vies": mult_vies,
                            "volume_base": volume_base,
                        }

                        preco_abertura = candles_feat.iloc[-1]["close"]
                        timestamp_execucao = datetime.now()
                        timestamp_inicio_candle = candles_feat.iloc[-1].get("timestamp", timestamp_execucao)
                        if isinstance(timestamp_inicio_candle, str):
                            try:
                                timestamp_inicio_candle = datetime.fromisoformat(timestamp_inicio_candle)
                            except Exception:
                                timestamp_inicio_candle = timestamp_execucao

                        # Delay de execução
                        log_delay_execucao(
                            timestamp_inicio_candle=timestamp_inicio_candle,
                            timestamp_execucao_ordem=timestamp_execucao,
                            ativo=ativo,
                            ciclo=ciclo_num,
                            info_extra=f"sinal={sinal_traduzido}, volume=auto"
                        )

                        # Envio da ordem
                        resultado = abrir_ordem_e_registrar(
                            ativo=ativo,
                            tipo=sinal_traduzido,
                            volume=volume_final,  # volume adaptativo * multiplicador de viés
                            timestamp=candles_feat.iloc[-1]["timestamp"],
                            preco_abertura=preco_abertura,
                            contexto=ctx_exec,
                            observacao=resultado_sinal.get("motivo", "")
                        )

                        # volume efetivamente enviado (ou 'auto' se não vier no diagnóstico)
                        volume_usado = _extrair_volume_usado(resultado, ativo)
                        volume_field = volume_usado if volume_usado is not None else "auto"

                        ticket = resultado.get("order")
                        retcode = resultado.get("retcode")
                        comment = resultado.get("comment", "")
                        diag = resultado.get("diagnostico", None)

                        ciclo_info_log[f"{ativo}_ordem"] = {
                            "ticket": ticket,
                            "retcode": retcode,
                            "sinal": sinal_traduzido,
                            "volume": volume_field,
                            "volume_base": volume_base,
                            "mult_vies": mult_vies,
                            "timestamp": candles_feat.iloc[-1]["timestamp"],
                            "preco_abertura": preco_abertura,
                            "padrao": padrao,
                            "score_total": score_total_ajust,
                            "comment": comment,
                        }

                        campos_essenciais = ["ticket", "retcode", "sinal", "volume", "preco_abertura", "padrao", "score_total"]
                        problemas = sanity_check_dict(ciclo_info_log[f"{ativo}_ordem"], campos_essenciais)
                        if problemas:
                            ciclo_info_log[f"{ativo}_ordem"]["problemas"] = ";".join(problemas)
                            log_event(f"[SANITY CHECK] Dados incompletos na ordem: {problemas}", level="warning")

                        if ticket and str(ticket).isdigit() and int(ticket) > 0 and str(retcode).startswith("100"):
                            operou = True
                            log_event(
                                f"[ORDEM OK] ativo={ativo} ticket={ticket} retcode={retcode} volume={volume_field} "
                                f"sinal={sinal_traduzido} preco={preco_abertura}",
                                level="info"
                            )
                            try:
                                registrar_contexto_ordem(
                                    ordem_info={
                                        "ativo": ativo,
                                        "timestamp": candles_feat.iloc[-1]["timestamp"],
                                        "padrao": padrao,
                                        "tipo": sinal_traduzido,
                                        "volume": volume_field,
                                        "volume_base": volume_base,
                                        "mult_vies": mult_vies,
                                        "retcode": retcode,
                                        "ticket": ticket,
                                        "lucro": 0
                                    },
                                    contexto_info={
                                        "regime": regime,
                                        "contexto": contexto,
                                        "hora": time.strftime("%H:%M:%S")
                                    }
                                )
                            except Exception as e:
                                log_event(f"[CONTEXTO] Falha ao registrar contexto institucional: {e}", level="error")
                        else:
                            motivo_fail = resultado.get("motivo")
                            log_event(
                                f"Ordem NÃO registrada para {ativo} "
                                f"(Ticket: {ticket}, Retcode: {retcode}, Motivo: {motivo_fail}, Comentário: {comment}) "
                                f"(CICLO {ciclo_num})",
                                level="warning"
                            )
                            # Diagnóstico detalhado do MT5
                            if isinstance(diag, dict):
                                try:
                                    _log_bloqueio(
                                        ativo, ciclo_num, "mt5_retcode",
                                        retcode=retcode, motivo=motivo_fail, comment=comment,
                                        last_error=diag.get("last_error"),
                                        symbol_info=diag.get("symbol_info")
                                    )

                                    log_event(
                                        f"[MT5/DIAG] {ativo} | last_error={diag.get('last_error')} | "
                                        f"symbol_info={diag.get('symbol_info')}",
                                        level="warning"
                                    )
                                except Exception:
                                    pass
                            else:
                                # tenta coletar algo do MT5 mesmo assim
                                try:
                                    last_err = mt5.last_error()
                                    si = mt5.symbol_info(ativo)
                                    si_dict = None
                                    if si:
                                        si_dict = {
                                            "trade_mode": getattr(si, "trade_mode", None),
                                            "digits": getattr(si, "digits", None),
                                            "spread": getattr(si, "spread", None),
                                            "stops_level": getattr(si, "stops_level", None),
                                            "freeze_level": getattr(si, "freeze_level", None),
                                            "volume_min": getattr(si, "volume_min", None),
                                            "volume_max": getattr(si, "volume_max", None),
                                            "volume_step": getattr(si, "volume_step", None),
                                            "filling_mode": getattr(si, "filling_mode", None),
                                        }
                                    _log_bloqueio(
                                        ativo, ciclo_num, "mt5_retcode",
                                        retcode=retcode, comment=comment, last_error=last_err, symbol_info=si_dict
                                    )
                                except Exception:
                                    pass

                        # --- Memória adaptativa ---
                        try:
                            reforcar_memoria(ativo, resultado_sinal, retcode)
                            log_event(f"Memória adaptativa reforçada para {ativo} (CICLO {ciclo_num})", level="info")
                        except Exception as e:
                            log_event(f"Erro ao reforçar memória para {ativo}: {e} (CICLO {ciclo_num})", level="warning")

                    except Exception as e:
                        log_event(f"Falha no processamento do sinal para {ativo}: {e}", level="error")
                        ciclo_info_log[f"{ativo}_ordem"] = {"erro": str(e)}

                # --- Comandos via Telegram ---
                try:
                    checar_comando()
                    comando_telegram = ler_comando_telegram()
                    if comando_telegram == "fechar_todas":
                        total_fechadas = fechar_todas_ordens(ativos, motivo="Fechamento manual via Telegram")
                        setar_comando_telegram("")  # Limpa o comando para não executar novamente
                        enviar_telegram(f"✅ Todas as ordens ({total_fechadas}) foram fechadas via comando Telegram.")
                        if total_fechadas > 0:
                            operou = True
                except Exception as e:
                    log_event(f"Erro ao executar comando Telegram 'fechar_todas': {e}", level="error")

                if not operou:
                    log_event("[MAIN LOOP] Nenhuma operação. Aguardando próximo ciclo...", level="debug")

                gravar_ultimo_ciclo_log(ciclo_info_log)

            except Exception as e:
                log_event(f"[ERRO CRÍTICO] Exceção não tratada no ciclo {ciclo_num}: {e}", level="error")
                time.sleep(30)
            finally:
                try:
                    tc.finalizar_ciclo()
                    try:
                        tc.logar(log_event, ciclo=ciclo_num)
                    except Exception:
                        pass
                    try:
                        if config.get("telemetria_tempo_ciclo_csv", False):
                            tc.salvar_csv(ciclo=ciclo_num)
                    except Exception:
                        pass
                except Exception:
                    pass

                # Medição confiável do tempo de ciclo
                try:
                    dur = time.perf_counter() - ciclo_t0
                    log_event(f"[CICLO] Duração: {dur:.3f} s (ciclo {ciclo_num})", level="info")
                except Exception as _e:
                    log_event(f"[CICLO] Falha ao medir duração do ciclo {ciclo_num}: {_e}", level="warning")

    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n[!] Execução interrompida pelo usuário. Robô parado com segurança." + Style.RESET_ALL)
        log_event("[STOP] Execução interrompida pelo usuário.", level="warning")
        try:
            enviar_telegram("🛑 Robô FTMO INSTITUCIONAL interrompido manualmente.")
        except Exception:
            pass

if __name__ == "__main__":
    main()
