import sys
from mt5.executar_ordem_mt5 import enviar_ordem
from utils.debug_logger import log_event

def testar_ordem_mt5():
    ativo = "EURUSD"
    direcao = "compra"  # ou "venda"
    volume = 0.1
    timestamp = "2025-07-15 03:00:00"  # pode ser datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\nEnviando ordem de TESTE: {direcao.upper()} {ativo}, volume={volume}...")
    log_event(f"[TESTE_MT5] Enviando ordem teste: {direcao} {ativo}, volume={volume}")

    try:
        resultado = enviar_ordem(direcao, ativo, timestamp, volume=volume)
        print("\nResultado da execução:")
        print(resultado)
        log_event(f"[TESTE_MT5] Resultado da ordem: {resultado}")
        if resultado.get("order") and str(resultado.get("order")).isdigit():
            print(f"✅ Ordem enviada com sucesso! Bilhete: {resultado['order']}")
        else:
            print("❌ Ordem NÃO enviada! Veja detalhes acima.")
    except Exception as e:
        print(f"❌ Erro ao enviar ordem para MT5: {e}")
        log_event(f"[TESTE_MT5] Erro ao enviar ordem: {e}", level="error")

if __name__ == "__main__":
    testar_ordem_mt5()
