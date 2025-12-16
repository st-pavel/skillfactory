####----------- 4.9 ------------
ratings = [('Old York', 3.3), ('New Age', 4.6), ('Old Gold', 3.3), ('General Foods', 4.8),
           ('Belissimo', 4.5), ('CakeAndCoffee', 4.2), ('CakeOClock', 4.2), ('CakeTime', 4.1),
           ('WokToWork', 4.9), ('WokAndRice', 4.9), ('Old Wine Cellar', 3.3), ('Nice Cakes', 3.9)]
from collections import OrderedDict

cafes = OrderedDict(sorted(sorted(ratings, key = lambda x : x[0]), key = lambda x : x[1], reverse = True))

print(cafes)