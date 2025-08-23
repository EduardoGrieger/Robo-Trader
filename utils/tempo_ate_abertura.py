from datetime import datetime, timedelta, timezone

def tempo_ate_abertura_utc():
    agora = datetime.now(timezone.utc)

    # Sexta-feira às 21:00 UTC fecha o mercado
    if agora.weekday() == 4 and agora.hour >= 21:
        # Próxima abertura é domingo 22:00 UTC
        dias_ate_domingo = (6 - agora.weekday())  # 6 = domingo
        proxima_abertura = (agora + timedelta(days=dias_ate_domingo)).replace(hour=22, minute=0, second=0, microsecond=0)
        if proxima_abertura <= agora:
            proxima_abertura += timedelta(days=7)
        return proxima_abertura - agora

    # Sábado (todo dia fechado até domingo)
    if agora.weekday() == 5:
        dias_ate_domingo = (6 - agora.weekday())
        proxima_abertura = (agora + timedelta(days=dias_ate_domingo)).replace(hour=22, minute=0, second=0, microsecond=0)
        return proxima_abertura - agora

    # Domingo antes das 22:00 UTC mercado fechado
    if agora.weekday() == 6 and agora.hour < 22:
        proxima_abertura = agora.replace(hour=22, minute=0, second=0, microsecond=0)
        return proxima_abertura - agora

    # Mercado aberto
    return timedelta(0)
