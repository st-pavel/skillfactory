# Импортируем объект Counter из модуля collections
from collections import Counter

# Создаём пустой объект Counter
c = Counter()

cars = ['red', 'blue', 'black', 'black', 'black', 'red', 'blue', 'red', 'white']
c = Counter(cars)
print(c)
print(c['black'])
print(sum(c.values()))
print(c.values())


def multiplier(x):
    print(x)
    def inner(y): 
        print(x, y)
        def inner2(z):
            print(x, y, z)
            return x * y + z    
        
        return inner2
    return inner
print(multiplier(2)(5)(9)) 