def exchange(usd= None, rub= None, rate= None):
  temp = 0
  #print(f'USD = {usd}, RUB = {rub}, Rate = {rate}')
  arg = [usd, rub, rate]
  if arg.count(None) >= 2 : raise ValueError('Not enough arguments')
  if arg.count(None) == 0 : raise ValueError('Too many arguments')
  if usd != None and rub != None:
      temp = rub/usd
  if usd != None and rate != None:
      temp = usd*rate
  if rate != None and rub != None:
      temp = rub/rate
  exchange = temp
  return exchange

print(exchange(usd = 12))

#print(exchange(usd=100, rub=8500))
# 85.0

#print(exchange(usd=100, rate=85))
# 8500

#print(exchange(rub=1000, rate=85))
# 11.764705882352942

#print(exchange(rub=1000, rate=85, usd=90))
# ValueError: Too many arguments

#print(exchange(rub=1000))
# ValueError: Not enough arguments