# В переменной number хранится некоторое число.
# Напишите программу со следующим условием: если число number не ноль,
# то программа делит 10 на введённое число и записывает результат,
# округлённый до трёх знаков после запятой, в переменную result.
# Если же в переменной number хранится ноль, то программа должна
# выбросить исключение ZeroDivisionError с фразой
# "Вы собираетесь делить на 0" с помощью ключевого слова raise.
# print('ZeroDivisionError: Вы собираетесь делить на 0')


number = 0
result = 0

try:
    if number == 0:
        raise ZeroDivisionError("Вы собираетесь делить на 0")
    result = round(10 / number, 3)
except ZeroDivisionError as e:
    result = None
    print(e)

print(result)
