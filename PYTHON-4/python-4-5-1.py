# Список вещей, которые нужно положить в инвентарь.
to_inventory = ['Blood Moon Sword', 'Sunset-colored sword', 'Bow of Stars', 'Gain Stone']
# Задаём пустой инвентарь
inventory = [] 
# Создаём цикл по элементам списка to_inventory
# item — текущий элемент списка
for item in to_inventory: 
    # Проверяем условие, что инвентарь уже заполнен.
    if len(inventory) == 3: 
        # Если условие выполняется,выводим предупреждение об ошибке.
        print('Inventory is full!') 
        # Завершаем работу цикла
        break 
    # Если цикл не завершился добавляем предмет в инвентарь
    inventory.append(item)
# Выводим результирующий инвентарь 
print(inventory)