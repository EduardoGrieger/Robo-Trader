import streamlit as st
import pandas as pd
import duckdb
import os
from io import StringIO
import glob
import json
from datetime import datetime
import plotly.express as px

# Configurações iniciais da página
st.set_page_config(page_title="Painel Robô Institucional FTMO", layout="wide")
st.title("🤖 Painel Institucional do Robô Trader — FTMO PRO")

# === Paths principais ===
DB_PATH = "dados/robodados.duckdb"
RANKING_PATH = "dados/ranking_padroes.csv"
MEMORIA_PATH = "dados/memoria_adaptativa.csv"
DELAY_PATH = "logs/monitor_delay_execucao.csv"
CONFIG_PATH = "config.json"
LOGS_DIR = "logs/"

# === Funções utilitárias ===

def carregar_operacoes():
    try:
        con = duckdb.connect(DB_PATH)
        df = con.execute("SELECT * FROM operacoes ORDER BY timestamp ASC").fetchdf()
        con.close()
        return df
    except Exception:
        return pd.DataFrame()

def carregar_delay():
    try:
        return pd.read_csv(DELAY_PATH)
    except Exception:
        return pd.DataFrame()

def carregar_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def status_robo():
    log_files = sorted(glob.glob(os.path.join(LOGS_DIR, "debug_*.log")), reverse=True)
    if os.path.exists("robo.lock"):
        return ("⏸️ Pausado (lock ativo)", "orange")
    if log_files:
        try:
            with open(log_files[0], "r", encoding="utf-8") as f:
                lines = f.readlines()[-100:]
            for line in reversed(lines):
                if "Robô iniciado" in line:
                    return ("🟢 Rodando", "green")
                if "Execução interrompida" in line:
                    return ("🔴 Parado", "red")
        except Exception:
            pass
    return ("⚪ Status Desconhecido", "gray")

def status_mt5():
    try:
        import MetaTrader5 as mt5
        return "🟢 Conectado" if mt5.initialize() else "🔴 Desconectado"
    except Exception:
        return "❌ MetaTrader5 não instalado"

def carregar_ordens_abertas_mt5():
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return pd.DataFrame()
        posicoes = mt5.positions_get()
        if not posicoes:
            return pd.DataFrame()
        df = pd.DataFrame([p._asdict() for p in posicoes])
        return df
    except Exception:
        return pd.DataFrame()

def colorir_lucro(val):
    try:
        val = float(val)
        if val > 0:
            return "color: #2196f3; font-weight: bold;"  # azul
        elif val < 0:
            return "color: #e53935; font-weight: bold;"  # vermelho
    except Exception:
        pass
    return ""

def filtrar_por_data(df, coluna_data, data_inicio, data_fim):
    mask = (pd.to_datetime(df[coluna_data]) >= data_inicio) & (pd.to_datetime(df[coluna_data]) <= data_fim)
    return df.loc[mask]

# === Layout em abas ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Status & Performance",
    "Ordens",
    "Ranking de Padrões",
    "Memória Adaptativa",
    "Logs",
    "Delay Execução",
    "Configuração"
])

# --- TAB 1: STATUS & PERFORMANCE ---
with tab1:
    st.header("Status Geral e Performance do Robô")

    status_text, status_color = status_robo()
    status_mt5_text = status_mt5()

    col1, col2 = st.columns([1,1])
    col1.metric("Status Robô", status_text, delta_color=status_color)
    col2.metric("Status MetaTrader 5", status_mt5_text)

    df_ops = carregar_operacoes()

    if df_ops.empty:
        st.info("Nenhuma operação registrada no banco de dados.")
    else:
        # Filtro de período
        st.subheader("Filtros de Visualização")
        col_data_ini, col_data_fim = st.columns(2)
        data_min = pd.to_datetime(df_ops['timestamp']).min()
        data_max = pd.to_datetime(df_ops['timestamp']).max()
        data_inicio = col_data_ini.date_input("Data Início", data_min.date(), min_value=data_min.date(), max_value=data_max.date())
        data_fim = col_data_fim.date_input("Data Fim", data_max.date(), min_value=data_min.date(), max_value=data_max.date())

        df_ops_filtered = filtrar_por_data(df_ops, 'timestamp', pd.to_datetime(data_inicio), pd.to_datetime(data_fim) + pd.Timedelta(days=1))

        lucro_total = df_ops_filtered['resultado'].sum()
        drawdown = df_ops_filtered['resultado'].cumsum().min()
        acertos = (df_ops_filtered['resultado'] > 0).sum()
        erros = (df_ops_filtered['resultado'] < 0).sum()
        neutros = (df_ops_filtered['resultado'] == 0).sum()
        taxa_acerto = acertos / max(1, len(df_ops_filtered))

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Lucro Total (USD)", f"${lucro_total:.2f}")
        colB.metric("Drawdown Máximo (USD)", f"${drawdown:.2f}")
        colC.metric("Taxa de Acerto", f"{taxa_acerto*100:.1f}%")
        colD.metric("Acertos / Erros / Neutros", f"{acertos} / {erros} / {neutros}")

        # Gráfico lucro acumulado com Plotly para zoom/hover
        fig_lucro = px.line(
            df_ops_filtered.assign(acum_lucro=df_ops_filtered['resultado'].cumsum()),
            x='timestamp', y='acum_lucro',
            title='Lucro Acumulado ao Longo do Tempo',
            labels={'timestamp': 'Data', 'acum_lucro': 'Lucro Acumulado (USD)'}
        )
        st.plotly_chart(fig_lucro, use_container_width=True)

        # Gráfico barras dos resultados por operação
        fig_barras = px.bar(
            df_ops_filtered,
            x='timestamp',
            y='resultado',
            color='resultado',
            color_continuous_scale=['red', 'gray', 'blue'],
            title='Resultado por Operação',
            labels={'timestamp': 'Data', 'resultado': 'Resultado (USD)'}
        )
        st.plotly_chart(fig_barras, use_container_width=True)

        # Download CSV filtrado
        csv_buffer = StringIO()
        df_ops_filtered.to_csv(csv_buffer, index=False)
        st.download_button("📥 Baixar Operações Filtradas (CSV)", csv_buffer.getvalue(), "operacoes_filtradas.csv", mime="text/csv")

# --- TAB 2: ORDENS ---
with tab2:
    st.header("Ordens Abertas (MT5 Tempo Real) e Fechadas (Histórico)")

    df_abertas_mt5 = carregar_ordens_abertas_mt5()
    st.subheader("📂 Ordens Abertas (MT5 - Tempo Real)")

    if df_abertas_mt5.empty:
        st.info("Nenhuma ordem aberta detectada no MetaTrader 5.")
    else:
        df_show = df_abertas_mt5[[
            "ticket", "symbol", "type", "volume", "price_open", "profit", "time"
        ]].rename(columns={
            "ticket": "Bilhete",
            "symbol": "Ativo",
            "type": "Tipo",
            "volume": "Lote",
            "price_open": "Preço Abertura",
            "profit": "Lucro Atual",
            "time": "Timestamp"
        })
        styled = df_show.style.applymap(colorir_lucro, subset=["Lucro Atual"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("📁 Ordens Fechadas (Banco de Dados)")
    if df_ops.empty:
        st.info("Nenhuma ordem fechada registrada no banco.")
    else:
        # Filtro por ativo e período
        ativos_disponiveis = df_ops['ativo'].unique()
        ativo_selecionado = st.selectbox("Filtrar por Ativo", options=["Todos"] + list(ativos_disponiveis), index=0)

        if ativo_selecionado != "Todos":
            df_filtered = df_ops[df_ops['ativo'] == ativo_selecionado]
        else:
            df_filtered = df_ops

        data_min = pd.to_datetime(df_filtered['timestamp']).min()
        data_max = pd.to_datetime(df_filtered['timestamp']).max()
        data_inicio = st.date_input("Data Início (Fechadas)", data_min.date(), min_value=data_min.date(), max_value=data_max.date(), key="fechadas_ini")
        data_fim = st.date_input("Data Fim (Fechadas)", data_max.date(), min_value=data_min.date(), max_value=data_max.date(), key="fechadas_fim")

        df_filtered = filtrar_por_data(df_filtered, 'timestamp', pd.to_datetime(data_inicio), pd.to_datetime(data_fim) + pd.Timedelta(days=1))
        ordens_fechadas = df_filtered[df_filtered['data_fechamento'].notna()].sort_values("data_fechamento", ascending=False)
        st.dataframe(ordens_fechadas.head(100), use_container_width=True, hide_index=True)

        # Download CSV das ordens fechadas filtradas
        csv_buffer = StringIO()
        ordens_fechadas.to_csv(csv_buffer, index=False)
        st.download_button("📥 Baixar Ordens Fechadas Filtradas (CSV)", csv_buffer.getvalue(), "ordens_fechadas_filtradas.csv", mime="text/csv")

# --- TAB 3: RANKING DE PADRÕES ---
with tab3:
    st.header("🏆 Ranking dos Padrões Técnicos")

    if os.path.exists(RANKING_PATH):
        df_ranking = pd.read_csv(RANKING_PATH)
        if {'acertos', 'total'}.issubset(df_ranking.columns):
            df_ranking['taxa_acerto'] = df_ranking['acertos'] / df_ranking['total']
        st.dataframe(df_ranking.sort_values("taxa_acerto", ascending=False).head(20), use_container_width=True, hide_index=True)
        csv_buffer = StringIO()
        df_ranking.to_csv(csv_buffer, index=False)
        st.download_button("📥 Baixar Ranking Completo (CSV)", csv_buffer.getvalue(), "ranking_padroes.csv", mime="text/csv")
    else:
        st.info("Ranking de padrões não disponível.")

# --- TAB 4: MEMÓRIA ADAPTATIVA ---
with tab4:
    st.header("🧠 Memória Adaptativa — Reforço de Padrões")

    if os.path.exists(MEMORIA_PATH):
        df_memoria = pd.read_csv(MEMORIA_PATH)
        st.dataframe(df_memoria.sort_values("score", ascending=False).head(20), use_container_width=True, hide_index=True)
        csv_buffer = StringIO()
        df_memoria.to_csv(csv_buffer, index=False)
        st.download_button("📥 Baixar Memória Adaptativa Completa (CSV)", csv_buffer.getvalue(), "memoria_adaptativa.csv", mime="text/csv")
    else:
        st.info("Dados de memória adaptativa não disponíveis.")

# --- TAB 5: LOGS ---
with tab5:
    st.header("📜 Logs Institucionais — Últimos 100 Eventos (Debug e Erros)")

    hoje_str = datetime.now().strftime("%Y-%m-%d")
    debug_log_path = os.path.join(LOGS_DIR, f"debug_{hoje_str}.log")
    erros_log_path = os.path.join(LOGS_DIR, f"erros_{hoje_str}.log")

    logs_mostrados = False

    if os.path.exists(debug_log_path):
        st.subheader(f"Debug ({os.path.basename(debug_log_path)})")
        with open(debug_log_path, "r", encoding="utf-8") as f:
            linhas = f.readlines()
        st.text("".join(linhas[-100:]))
        logs_mostrados = True
    else:
        st.warning("Arquivo de log debug do dia não encontrado.")

    if os.path.exists(erros_log_path):
        st.subheader(f"Erros ({os.path.basename(erros_log_path)})")
        with open(erros_log_path, "r", encoding="utf-8") as f:
            linhas = f.readlines()
        st.text("".join(linhas[-100:]))
        logs_mostrados = True
    else:
        st.warning("Arquivo de log de erros do dia não encontrado.")

    if not logs_mostrados:
        st.info("Nenhum log institucional do dia encontrado.")

# --- TAB 6: DELAY DE EXECUÇÃO ---
with tab6:
    st.header("⏱️ Delay de Execução das Ordens")

    df_delay = carregar_delay()
    if df_delay.empty:
        st.warning("Nenhum dado de delay encontrado.")
    else:
        df_delay["delay_segundos"] = pd.to_numeric(df_delay["delay_segundos"], errors="coerce")
        st.dataframe(df_delay.tail(100), use_container_width=True)

        fig_delay = px.line(
            df_delay,
            x="timestamp",
            y="delay_segundos",
            title="Delay de Execução por Timestamp",
            labels={"timestamp": "Timestamp", "delay_segundos": "Delay (s)"}
        )
        st.plotly_chart(fig_delay, use_container_width=True)

        st.metric("Média Delay Execução (s)", f"{df_delay['delay_segundos'].mean():.3f}")
        st.metric("Máximo Delay Execução (s)", f"{df_delay['delay_segundos'].max():.3f}")
        st.metric("Mínimo Delay Execução (s)", f"{df_delay['delay_segundos'].min():.3f}")

        csv_buffer = StringIO()
        df_delay.to_csv(csv_buffer, index=False)
        st.download_button("📥 Baixar Dados de Delay (CSV)", csv_buffer.getvalue(), "delay_execucao.csv", mime="text/csv")

# --- TAB 7: CONFIGURAÇÃO ---
with tab7:
    st.header("⚙️ Configuração Atual do Robô")

    config = carregar_config()
    st.json(config)

    # Permite editar configuração simples diretamente no painel e salvar
    with st.expander("Editar Configuração (JSON)"):
        config_edit = st.text_area("JSON Configuração", value=json.dumps(config, indent=2), height=300)
        if st.button("Salvar Configuração"):
            try:
                config_obj = json.loads(config_edit)
                with open(CONFIG_PATH, "w") as f:
                    json.dump(config_obj, f, indent=2)
                st.success("Configuração salva com sucesso! Reinicie o robô para aplicar.")
            except Exception as e:
                st.error(f"Erro ao salvar configuração: {e}")

st.caption("Robô Trader Institucional | Versão PRO | FTMO Ready 🚀 | v1.0")
