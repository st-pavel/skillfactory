# # Заданный список динамики числа пользователей
# user_dynamics = [-5, 2, 4, 8, 12, -7, 5]
# # Создаём цикл по индексам и элементам списка
# # i — индекс текущего элемента, dynamic — текущее значение из списка
# for i, dynamic in enumerate(user_dynamics):
#     # Выводим номер дня и динамику на этот день
#     print("Day {} : {}".format(i+1, dynamic))
#
#
# ## Будет выведено:
# ## Day 1 : -5
# ## Day 2 : 2
# ## Day 3 : 4
# ## Day 4 : 8
# ## Day 5 : 12
# ## Day 6 : -7
# ## Day 7 : 5


# ЗАДАНИЕ 5.3 (НА САМОПРОВЕРКУ)
#
# В нашем распоряжении всё та же информация о динамике пользователей user_dynamics в нашем приложении.
#
# Мы дали начинающему программисту задание написать программу, которая находит номер всех дней,
# когда наблюдался отток клиентов (динамика была отрицательной), а также находит количество
# ушедших в эти дни клиентов. Он написал следующий код:
#
# user_dynamics = [-5, 2, 4, 8, 12, -7, 5]
# number_negative = None #объявляем переменную, в которой будем хранить номер последнего дня оттока, изначально она пустая (None)
# #создаём цикл по элементам последовательности от 0 до N (не включая N)
# for i, dynamic in enumerate(user_dynamics):
#     #проверяем условие оттока — текущий элемент отрицательный
#     if dynamic < 0: #если условие истинно,
#         print("Churn value: ", dynamic) # выводим количество ушедших в этот день пользователей
#         print("Number day: ", i+1) # выводим номер дня
# #
#

# # Задание 5.4 (External resource)
# # Дан список, состоящий из строк. Список хранится в переменной str_list.
# # Создайте новый список cut_str_list.
# #
# # В цикле добавляйте в список cut_str_list списки, состоящие из порядкового
# # номера строки из списка str_list (номер в рамках данной задачи начинается с 0)
# # и трёх первых символов из каждой строки. Если длина строки меньше трёх
# # символов, то добавляется вся строка целиком.
#
# str_list = ['Hello', 'my', 'name', 'is', 'Ezeikel', 'I', 'like', 'knitting']
#
# ## cut_str_list = [[0, 'Hel'], [1, 'my'], [2, 'nam'], [3, 'is'], [4, 'Eze'], [5, 'I'], [6, 'lik'], [7, 'kni']]
#
# cut_str_list=[]
# for i, cur_string in enumerate(str_list):
#     cut_str_list.append([i,cur_string[:3]])
# print(cut_str_list)


# n = 27
# # Создаём бесконечный цикл
# while True:
#     # Проверяем условие, что остаток от деления на 3 равен 0.
#     if n % 3 == 0:
#         # Если условие выполняется, новое число — результат целочисленного деления на 3.
#         n = n // 3
#         # Проверяем условие, что в результате деления получили 1.
#         if n == 1:
#             # Выводим утвердительное сообщение
#             print('n - is the power of the number 3!')
#             # Выходим из цикла
#             break
#     else:
#         # В противном случае выводим сообщение-опровержение
#         print('n - is not the power of the number 3!')
#         # Выходим из цикла
#         break
# ## Будет выведено:
# ## n - is the power of the number 3!
#
# #
# # Задание 5.7 (External resource)
# # Допишите программу, которая проверяет гипотезу Сиракуз.
# #
# # Гипотеза Сиракуз заключается в том, что любое натуральное число n можно свести к 1, если повторять над ним следующие действия:
# #
# # если число чётное, разделить его нацело пополам, т. е. n = n // 2;
# # если нечётное — умножить на 3, прибавить 1 и результат нацело разделить на 2, т. е. n = (n * 3 + 1) // 2.
# # Мы тестируем только целые числа!
# n = 19
# ## Syracuse hypothesis holds for number 19
#
# n = 25921
# ## Syracuse hypothesis holds for number 25921
# origin = n
# # Создаём бесконечный цикл
# while True:
#     if n % 2 == 0:
#         print('Четное n =', n)
#         n = n // 2
#     else:
#         print('Нечетное n =', n)
#         n = (n * 3 + 1) // 2
#     print('Новое n=', n)
#     # Проверяем, что результат равен 1
#     if n == 1:
#         # Если условие выполняется, выводим утвердительное сообщение.
#         print(f'Syracuse hypothesis holds for number {origin}')
#         break
#
#         # Ваш код здесь
#


# #
# # Заданный словарь
# client_status = {
#     103303: 'yes',
#     103044: 'no',
#     100423: 'yes',
#     103032: 'no',
#     103902: 'no'
# }
# # Создаём цикл по ключам словаря client_status
# for user_id in client_status:  # user_id — текущий ключ словаря
#
#     if client_status[user_id] == 'no':
#         # Если текущий статус == 'no', переходим на следующую итерацию
#         continue
#     else:
#         # В противном случае выводим сообщение об отправке сертификата
#         print('Send present user', user_id)
#
# # Будет выведено:
# # Send present user 103303
# # Send present user 100423

# # Задание 5.10 (External resource)
# #Дан словарь mixture_dict. Значения в нём могут быть трёх типов: строки (str) или числа (int и float).
# #Посчитайте, сколько значений в словаре mixture_dict являются числами. Результат занесите в
# # переменную count_numbers. Используйте в своём коде оператор continue.
# #
# #
# mixture_dict = {'a': 15, 'b': 10.5, 'c': '15', 'd': 50, 'e': 15, 'f': '15'}
# ## count_numbers = 4
#
# #mixture_dict = {'key1': 24, 'key2': '1.4', 'key3': 14, 'key4': 16.24, 'key6': 124.2414, 'key7': 12.2}
# ## count_numbers = 5
#
# count_numbers=0
#
# for elem in mixture_dict:
#     if not isinstance(mixture_dict[elem], int) and not isinstance(mixture_dict[elem], float): # проще было бы проверять сразу на строку.
#         # Но тогда оператор continue не будет нужен, а его надо добавить по условиям задания
#         continue
#     else:
#         count_numbers += 1
# print(count_numbers)

# ### Задача. Условие задачи. Подсчитать количество вхождений каждого символа в заданном тексте.
# # В результате работы программы должен быть сформирован словарь, ключи которого —
# # символы текста, а значения — количество вхождений символа в тексте.
# text = """
# The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.
#
# Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next. First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs. She took down a jar from one of the shelves as she passed; it was labelled `ORANGE MARMALADE', but to her great disappointment it was empty: she did not like to drop the jar for fear of killing somebody, so managed to put it into one of the cupboards as she fell past it.
#
# `Well!' thought Alice to herself, `after such a fall as this, I shall think nothing of tumbling down stairs! How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell off the top of the house!' (Which was very likely true.)
# """
#
# # Приводим текст к нижнему регистру
# text = text.lower()
# # Заменяем пробелы на пустые строки
# text = text.replace(" ", "")
# # Заменяем символы переноса строки на пустые строки
# text = text.replace("\n", "")
# # Создаём пустой словарь для подсчёта количества символов
# count_dict = {}
# # Создаём цикл по символам в строке text
# for symbol in text:  # symbol — текущий символ в тексте
#     # Проверяем условие, что символа ещё нет среди ключей словаря.
#     if symbol not in count_dict:
#         # Если условие выполняется, заносим символ в словарь со значением 1.
#         count_dict[symbol] = 1  #
#     else:
#         # В противном случае увеличиваем частоту символа
#         count_dict[symbol] += 1
#     # Выводим результирующий словарь
# print(count_dict)
# ## {'t': 88, 'h': 63, 'e': 112, 'r': 39, 'a': 70, 'b': 15, 'i': 47, '-': 2, 'o': 80,
# # 'l': 55, 'w': 31, 'n': 57, 's': 60, 'g': 17, 'k': 14, 'u': 22, 'f': 28, 'm': 17, 'y': 18,
# # ',': 13, 'd': 43, 'p': 23, 'c': 9, 'v': 8, '.': 5, 'x': 1, ';': 3, 'j': 2,
# # '`': 3, "'": 5, ':': 1, '!': 4, '(': 1, ')': 1}

########################

# #Задание 5.11 (External resource)
# #Дана переменная text, в которой содержится некоторый текст. В этом тексте могут присутствовать знаки
# # препинания: точки ('.'), запятые(','), точки с запятой (';'), вопросительные ('?') и
# # восклицательные знаки ('!'), а также тире ('—').
#
# #Напишите программу, формирующую словарь count_punctuation, ключами которого являются
# # знаки препинания, а значениями — их количество в тексте.
#
# #Обратите внимание, что если какого-то из шести указанных знаков препинания нет в тексте,
# # то в итоговом словаре значение, соответствующее этому знаку препинания, должно быть равно 0.
# # Также имейте в виду, что тире и дефис являются разными символами.
#
# text = """
# She sells sea shells on the sea shore;
# The shells that she sells are sea shells I am sure.
# So if she sells sea shells on the sea shore,
# I am sure that the shells are sea shore shells.
# """
# ## count_punctuation = {',': 1, '.': 2, '?': 0, '!': 0, ';': 1, '—': 0}
#
# #text = """
# #Иной раз казалось ему, что он уже с месяц лежит; в другой раз — что всё тот же день идет. Но об том — об том он совершенно забыл; зато ежеминутно помнил, что об чем-то забыл, чего нельзя забывать, — терзался, мучился, припоминая, стонал, впадал в бешенство или в ужасный, невыносимый страх.
# #Тогда он порывался с места, хотел бежать, но всегда кто-нибудь его останавливал силой, и он опять впадал в бессилие и беспамятство. Наконец он совсем пришел в себя!
# #"""
# ### count_punctuation = {',': 12, '.': 3, '?': 0, '!': 1, ';': 2, '—': 3}
#
# count_punctuation={}
# punctuation = [',', '.', '?', '!', ';', '—']
# for punct in punctuation:
#     if punct not in count_punctuation:
#         count_punctuation[punct] = 0
#
# for symbol in text:
#     if symbol in punctuation:
#         count_punctuation[symbol] += 1
#
# print(count_punctuation)


#####################
# #Задание 5.12 (External resource)
# #Дано предложение, хранящееся в переменной sentence.
#
# #Выполните следующие действия:
#
# #Приведите все символы в предложении к нижнему регистру.
# #Исключите знаки препинания (запятые и точки).
# #Разделите строку по пробелам и создайте список слов, которые входят в предложение.
# #Подсчитайте количество повторений каждого слова.
# #Для подсчёта слов используйте словарь. Ключами словаря будут слова,
# # а значениями — число вхождений этих слов в предложение. В результате работы вашей программы
# # в переменной word_dict должен храниться результирующий словарь.
# #
#
# sentence = 'A roboT MAY Not injure a humAn BEING or, tHROugh INACtion, allow a human BEING to come to harm.'
# ## word_dict = {'a': 3, 'robot': 1, 'may': 1, 'not': 1, 'injure': 1, 'human': 2, 'being': 2, 'or': 1, 'through': 1, 'inaction': 1, 'allow': 1, 'to': 2, 'come': 1, 'harm': 1}
#
# #sentence = "СчастьЕ нЕ в тоМ, чтобы делать всегда, что хочешь, а в Том, чтОБЫ всегдА хотеть того, ЧТО дЕлаешь."
# ## word_dict = {'счастье': 1, 'не': 1, 'в': 2, 'том': 2, 'чтобы': 2, 'делать': 1, 'всегда': 2, 'что': 2, 'хочешь': 1, 'а': 1, 'хотеть': 1, 'того': 1, 'делаешь': 1}
#
# # Приводим текст к нижнему регистру
# sentence = sentence.lower()
# # # Заменяем символы переноса строки на пробелы
# # sentence = sentence.replace("\n", " ")
# # Заменяем запятые на пустые строки
# sentence = sentence.replace(",", "")
# # Заменяем точки на пустые строки
# sentence = sentence.replace(".", "")
# # # Заменяем точки с запятыми на пустые строки
# # sentence = sentence.replace(";", "")
# # Разделяем текст на слова
# word_sentence = sentence.split()
# # Создаём пустой словарь для подсчёта количества слов
# word_dict={}
# # Создаём цикл по словам в списке word_list
# # word — текущее слово из списка word_list
# for word in word_sentence:
#     # Проверяем условие, что слова ещё нет среди ключей словаря
#     if word not in word_dict:
#         # Если условие выполняется, заносим слово в словарь со значением 1
#         word_dict[word] = 1
#     else:
#         # В противном случае, увеличиваем частоту слова
#         word_dict[word] += 1
# print(word_dict)
#
# #########################
# #Задание 5.13 (External resource)
# #Дан список из строк str_list. Например:
# #str_list = ["text", "morning", "notepad", "television", "ornament"]
# #Напишите программу для подсчёта количества вхождений заданного символа в каждую из строк этого списка.
# # Искомый символ хранится в переменной symbol_to_check.
# #Для подсчёта используйте словарь: в качестве ключа запишите в него строку, в качестве значения — число
# # вхождений искомого символа в эту строку. Для хранения словаря используйте переменную с именем word_dict.
# #
# str_list = ["text", "morning", "notepad", "television", "ornament"]
# symbol_to_check = 't'
# ## word_dict = {'text': 2, 'morning': 0, 'notepad': 1, 'television': 1, 'ornament': 1}
#
# # str_list = ["text", "morning", "notepad", "television", "ornament"]
# # symbol_to_check = 'n'
# # ## word_dict = {'text': 0, 'morning': 2, 'notepad': 1, 'television': 1, 'ornament': 2}
#
# word_dict={}
# for row in str_list:
#     if row not in word_dict: # условиями задачи не оговорена уникальность слов в списке str_list, поэтому
#         #для решения исходим из их уникальности
#         word_dict[row] = 0
#     for string in row:
#         if symbol_to_check == string:
#             word_dict[row] += 1
# print(word_dict)
#


