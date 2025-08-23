def normalizar_sinal(sinal):
    """
    Converte sinal de qualquer formato para int padrão: 1 (compra), -1 (venda), 0 (neutro)
    """
    if isinstance(sinal, int):
        if sinal in [1, -1, 0]:
            return sinal
    s = str(sinal).strip().lower()
    if s in ['1', 'compra', 'comprar', 'buy', 'long']:
        return 1
    if s in ['-1', 'venda', 'vender', 'sell', 'short']:
        return -1
    if s in ['0', 'neutro', 'neutral', 'none', 'nan']:
        return 0
    # Default fallback:
    return 0

def sinal_to_str(sinal):
    """
    Converte sinal int para string amigável (ex: para logs)
    """
    s = normalizar_sinal(sinal)
    if s == 1:
        return "compra"
    if s == -1:
        return "venda"
    return "neutro"
