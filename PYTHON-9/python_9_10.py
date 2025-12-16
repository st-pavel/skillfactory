# Напишите функцию get_chess(a), которая принимает на вход длину стороны 
# квадрата a и возвращает двумерный массив формы (a, a), заполненный 0 и 1 
# в шахматном порядке. В левом верхнем углу всегда должен быть ноль.

# Примечание. воспользуйтесь функцией np.zeros, а затем с помощью срезов без циклов задайте необходимым элементам значение 1.

# В Python для получения каждого второго элемента используется срез [::2].
# Подумайте, как грамотно применить этот принцип к двумерному массиву.

import numpy as np
def get_chess(a):
    array = np.zeros((a, a))
    array[1::2, 0::2] = 1
    array[0::2, 1::2] = 1
    return array


print(get_chess(8))


# Задание 10.7 (External resource)
# Вы разрабатываете приложение для прослушивания музыки. Конечно же, 
# там будет доступна функция перемешивания плейлиста. Пользователю 
# может настолько понравиться перемешанная версия плейлиста, что он 
# захочет сохранить его копию. Однако вы не хотите хранить в памяти 
# новую версию плейлиста, а просто хотите сохранять тот seed, 
# с которым он был сгенерирован.

# Для этого напишите функцию shuffle_seed(array), которая принимает 
# на вход массив из чисел, генерирует случайное число для seed в 
# диапазоне от 0 до 2**32 - 1 (включительно) и возвращает кортеж: 
# перемешанный с данным seed массив (исходный массив должен оставаться 
# без изменений), а также seed, с которым этот массив был получен.


def shuffle_seed(array):
    seed_max_number = np.uint32(2**32 - 1)
    seed_number = np.random.randint(0, seed_max_number, dtype=np.uint32)
    np.random.seed(seed_number)
    mix = np.random.permutation(array)
    return mix, seed_number 


array = [1, 2, 3, 4, 5]
shuffle_seed(array)
# (array([1, 3, 2, 4, 5]), 2332342819)
shuffle_seed(array)
# (array([4, 5, 2, 3, 1]), 4155165971)



# Задание 10.8 (External resource)
# Напишите функцию min_max_dist(*vectors), которая принимает на вход 
# неограниченное число векторов через запятую. Гарантируется, что все 
# векторы, которые передаются, одинаковой длины.

# Функция возвращает минимальное и максимальное расстояние между векторами в виде кортежа.

def min_max_dist(*vectors):
    norm_list = []
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            norm_list.append(np.linalg.norm(vectors[i]-vectors[j]))
    
    return min(norm_list), max(norm_list)

vec1 = np.array([1,2,3])
vec2 = np.array([4,5,6])
vec3 = np.array([7, 8, 9])


print(min_max_dist(vec1, vec2, vec3))
# (5.196152422706632, 10.392304845413264)




# Задание 10.9 (External resource)
# Напишите функцию any_normal, которая принимает на вход неограниченное 
# число векторов через запятую. Гарантируется, что все векторы, которые передаются, одинаковой длины.
# Функция возвращает True, если есть хотя бы одна пара перпендикулярных векторов. Иначе возвращает False.

def any_normal(*vectors):
    dot_list = []
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            dot_list.append(np.dot(vectors[i],vectors[j]))
    array = np.array(dot_list)
    zero_in_array = (array == 0)
    return bool(np.count_nonzero(zero_in_array))



vec1 = np.array([2, 1])
vec2 = np.array([-1, 2])
vec3 = np.array([3,4])
print(any_normal(vec1, vec2, vec3))
# True

# Задание 10.10 (External resource)
# Напишите функцию get_loto(num), генерирующую трёхмерный массив 
# случайных целых чисел от 1 до 100 (включительно). Это поля для игры в лото.

# Трёхмерный массив должен состоять из таблиц чисел формы 5х5, 
# то есть итоговая форма — (num, 5, 5).

# Функция возвращает полученный массив.

def get_loto(num):
    loto_array = np.random.randint(1, 101, size=(num, 5,5), dtype=int)
        
    for i in range(num):
        a = set()
        while len(a) < 5*5:
            a.add(np.random.randint(1, 101))
        loto_add = np.array(list(a))
        loto_add.shape = (5, 5)
        loto_array[i,:,:] = loto_add
        
    return loto_array

print(get_loto(3))
