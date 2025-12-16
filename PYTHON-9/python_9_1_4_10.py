####-------------- Задание 4.10 (External resource)
# Напишите функцию task_manager(tasks), которая принимает список задач tasks для нескольких серверов. 
# Каждый элемент списка состоит из кортежа (<номер задачи>, <название сервера>, <высокий приоритет задачи>).
# Функция должна создавать словарь и заполнять его задачами по следующему принципу: 
# название сервера — ключ, по которому хранится очередь задач для конкретного сервера.
# Если поступает задача без высокого приоритета (последний элемент кортежа — False), 
# добавить номер задачи в конец очереди. Если приоритет высокий, добавить номер в начало.

# Для словаря используйте defaultdict, для очереди — deque.

# Функция возвращает полученный словарь с задачами.

from collections import deque, defaultdict


def task_manager(tasks):
    task_dict = defaultdict(deque)
    for task in tasks:
        if task[2]:
            task_dict[task[1]].appendleft(task[0])
        else:
            task_dict[task[1]].append(task[0])
    return task_dict


tasks = [(36871, 'office', False),
(40690, 'office', False),
(35364, 'voltage', False),
(41667, 'voltage', True),
(33850, 'office', False)]

print(task_manager(tasks))
# defaultdict(, {'voltage': deque([41667, 35364]),
# 'office': deque([36871, 40690, 33850])})



