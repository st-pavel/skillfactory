medicines = {'Ибупрофен': 99, 'Эспумизан': 279, 'Пенталгин': 119}
name = 'Визин'
#name = 'Ибупрофен'

try:
    print(medicines[name])
except KeyError:
    print('Такого ключа в словаре нет')
