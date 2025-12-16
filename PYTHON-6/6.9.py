## Введите свое решение ниже
def get_total_score(data):
    updated_data = list(map(lambda x: (*x, x[1]+x[2]+x[3]), data))
    updated_data.sort(key=lambda a: a[4])
    return updated_data


data = [
    ('Amanda', 37, 78, 67),
    ('Patricia', 78, 93, 68),
    ('Marcos', 79, 67, 89),
    ('Dmitry', 67, 68, 100),
    ('Andrey', 100, 78, 76),
    ('Victoria', 93, 69, 96),
]

print(get_total_score(data))
# [('Amanda', 37, 78, 67, 182), ('Marcos', 79, 67, 89, 235), ('Dmitry', 67, 68, 100, 235), ('Patricia', 78, 93, 68, 239), ('Andrey', 100, 78, 76, 254), ('Victoria', 93, 69, 96, 258)]
