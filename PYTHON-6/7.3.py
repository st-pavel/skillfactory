## Введите свое решение ниже
def find_cheese_type(cheese_data, cheese_type):
    filter_func = lambda item : item[1][3] == cheese_type
    mapped_func = lambda item : item[0] 

    filter_by_type_list = list(filter(filter_func, cheese_data.items())) 
    filter_by_type_list = list(map(mapped_func, filter_by_type_list))
    return filter_by_type_list

cheese_data = {
    'чеддер': [370, 5000, 33, 'твердый'],
    'пармезан': [510, 4000, 29, 'твердый'],
    'гауда': [250, 3700, 27, 'полутвердый'],
    'эдам': [220, 10000, 30, 'полутвердый'],
    'горгонзола': [320, 3000, 32, 'полумягкий'],
    'рокфор': [340, 15000, 31, 'полумягкий'],
    'стилтон': [360, 7000, 35, 'полумягкий'],
    'камамбер': [250, 8000, 24, 'мягкий'],
    'бри': [310, 6500, 28, 'мягкий'],
}

find_cheese_type(cheese_data, cheese_type='мягкий')
print(find_cheese_type(cheese_data, cheese_type='мягкий'))
# ['камамбер', 'бри']

find_cheese_type(cheese_data, cheese_type='полумягкий')
print(find_cheese_type(cheese_data, cheese_type='полумягкий'))
# ['горгонзола', 'рокфор', 'стилтон']
