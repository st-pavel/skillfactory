player_1, player_2 = ('бумага', 'камень')

if player_1 == player_2:
    print('Ничья!')
elif player_1 == 'камень' and player_2 == 'ножницы':
    print('Первый игрок — победитель!')
elif player_1 == 'камень' and player_2 == 'бумага':
    print('Второй игрок — победитель!')
elif player_1 == 'ножницы' and player_2 == 'камень':
    print('Второй игрок — победитель!')
elif player_1 == 'ножницы' and player_2 == 'бумага':
    print('Первый игрок — победитель!')
elif player_1 == 'бумага' and player_2 == 'камень':
    print('Первый игрок — победитель!')
elif player_1 == 'бумага' and player_2 == 'ножницы':
    print('Второй игрок — победитель!')
