import MetaTrader5 as mt5

mt5.initialize()
symbol = "EURUSD"
mt5.market_book_add(symbol)
book = mt5.market_book_get(symbol)
print(book)  # Normalmente retorna None ou [] em brokers retail
mt5.shutdown()
