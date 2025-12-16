#num_list = [98, 24, 23, 12, 3]
#num_list = list(range(20,1,-2))
# n = 3
# p = 1
#
# for num in range(1, n+1):
#     print('*'*num)
#
#
# # Задаём список значений массы товаров
# weight_of_products = [10, 42.4, 240.1, 101.5, 98, 0.4, 0.3, 15]
#
#
# # Задаём максимальное значение веса груза
# max_weight = 100
# # Задаём начальный номер груза
# num = 1
# # Создаём цикл по элементам списка со значениями массы товаров
# # weight — текущее значение веса
# for weight in weight_of_products:
#     # Если текущий вес меньше максимального,
#     if weight < max_weight:
#         # выводим номер груза, его вес и отправляем его в легковую машину.
#         print('Product {}, weight: {} -passenger car'.format(num, weight))
#     else:
#         # В противном случае
#         # выводим номер груза, его вес и отправляем его в грузовую машину.
#         print('Product {}, weight: {} -truck'.format(num, weight))
#     # Увеличиваем значение номера груза на 1
#     num += 1
#
# my_list = [1]
# for i in range(10):
#     print(i,my_list[i], my_list[i] * 2)
#     my_list.append(my_list[i] * 2)
#
# print(my_list, len(my_list))

#
#
# num_list = [1, 10, 3, -5]
# num_list.sort()
# for i in range(len(num_list)):
#     print(f'element {i}: {num_list[i]}')


# num_list = list(range(0, 100, 3))
# num_list = [10, 4, 4, 21, 2, 0, -10]
# num_list = list(range(0, 81, 5))
# count_even = 0
# for num in num_list:
#      if num%2==0:
#          count_even +=1
#
# print(count_even)


# mixture_list = [True, 1, -10, 'hello', False, 'string_1', 123, 2.5, [1, 2], 'another']
# mixture_list = ['My', 'Simple', -10, True, 'False', 'string_1', False, 2.5, [1, 2], 'True']
# count_str = 0
#
# for i in range(len(mixture_list)):
# #    print(mixture_list[i], type(mixture_list[i]), str(type(mixture_list[i])).split()[1] )
#     if str(type(mixture_list[i])).split()[1]== "'str'>":
#         count_str +=1
# #print(count_str)




#word_list = ["My", "name", "is", "Sergei", "EOS", "I'm", "from", "Moscow", "EOS"]
word_list = ["Человек", "он", "умный", "EOS", "Но", "чтоб", "умно", "поступать", "одного", "ума", "мало", "EOS"]


text = ''
for i in range(len(word_list)):
    if word_list[i] != "EOS":
        text = text + word_list[i] + ' '
    else:
        text = text[: - 1] + '. '
if text[: - 2] =='. ':
    text = text[: - 1]

print(text)