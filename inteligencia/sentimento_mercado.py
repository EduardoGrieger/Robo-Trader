# inteligencia/sentimento_mercado.py
import datetime
from utils.debug_logger import log_event

def obter_sentimento_mercado(keywords=("USD","EURUSD"), ultimos_min=15):
    """
    Stub de sentimento: retorna (score, noticias). Substitua por API real quando quiser.
    """
    agora = datetime.datetime.utcnow()
    noticias = [
        {"hora": agora.strftime("%Y-%m-%d %H:%M"), "headline": "Payroll acima do esperado, dólar dispara", "impacto": 0.8},
        {"hora": agora.strftime("%Y-%m-%d %H:%M"), "headline": "ECB mantém juros inalterados", "impacto": 0.3},
        {"hora": agora.strftime("%Y-%m-%d %H:%M"), "headline": "Tensão geopolítica eleva volatilidade no forex", "impacto": 0.5},
    ]
    score = sum(n["impacto"] for n in noticias if any(k.lower() in n["headline"].lower() for k in keywords))
    log_event(f"[NEWS] score={score:.2f} itens={len(noticias)} keys={keywords}", level="info")
    return score, noticias

def logar_noticias(noticias):
    for n in noticias:
        log_event(f"[NEWS] {n['hora']}: {n['headline']} (imp: {n['impacto']})", level="info")
