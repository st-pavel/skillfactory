# Задаём вес входящего в лифт человека
# weight = 67
# # Задаём грузоподъёмность
# max_weight = 400
# # Задаём суммарный вес людей в лифте
# S = 0
# # Создаём цикл, который будет работать, пока S не превысит max_weight.
# while S < max_weight: # делай, пока…
#     # Увеличиваем суммарный вес
#     # Равносильно S = S + weight
#     S += weight
#     # Выводим значение суммарного веса после обновления
#     print('Current sum weight', S)
#
#
# # Отделяем промежуточный вывод от результата пустой строкой
# print()
# # Выводим итоговое значение перевеса
# print('Overweight {} kg'.format(S - max_weight))


# # Задание 3.2
# volume = 10
# add_volume = 3.3
#
# cost = 0
# while cost < volume:
#     cost += add_volume
#
# cost = cost-volume
# print(cost)
#


# # Создаём накопительную переменную, в которой будем считать сумму.
# S = 0
# # Задаём текущее натуральное число
# n = 1
#
# # Создаём цикл, который будет работать, пока сумма не превысит 500.
# while S < 500:  # делай, пока …
#     # Увеличиваем сумму, равносильно S = S + n
#     S += n
#     # Увеличиваем значение натурального числа
#     n += 1
#     # Выводим строку ожидания
#     print("Still counting...")
# # Отделяем промежуточный вывод от результата пустой строкой
# print()
# # Выводим результирующую сумму
# print("Sum is: ", S)
# # Выводим результирующее количество чисел
# print("Numbers total: ", n - 1)

# # Задание 3.3
# value = 10
# value = 159
# value = 1000
#
# number = 1
# while number**2 < value:
#     number += 1
# print(number)

# # Задание 3.4
# value = 10
# value = 159
# value = 1000
#
#
#
# number = 1
# while True:
#     if number**2 > value:
#         break
#     number += 1
#
# print(number)



#
# # Задание 3.8
# n = 3
# i = 0
# while i < n:
#     print('Hello')
#     i+=1
#

# # Задание 3.9
# n = 100
# n = 156
# n = 12
#
# x = 1
# while not x ** 2 % n == 0:
#     x += 1
# print(x)

# #Задание 3.10
#
# value = 1000
# value = 178
# value = 10
#
# p = 1
# num = 1
#
# while p<value:
#     num +=1
#     p *= num
# print(p)

# #Задание 3.11
# money = 1000
# target_money = 3000
#
# money = 8000
# target_money = 15000
#
# money = 1500
# target_money = 300
#
# #interest = 8
#
# year_count = 0
#
# while not target_money < money:
#     money += money*8/100
#     year_count+=1
#
# print(year_count, money)
#


#Задание 3.12
# health = 500
# damage = 80
#
# health = 3014
# damage = 174
#
# seconds_num=0
#
# while health >0:
#     health-= damage
#     seconds_num+=1
# print(seconds_num)


#Задание 3.13 Числе Фибоначчи
n = 9

a = 1
b = 1
i = 2
fibonacci_list = [1,1]
while i< n:
    b = a + b
    a = b - a
    i+=1
    fibonacci_list.append(b)

print(fibonacci_list)


