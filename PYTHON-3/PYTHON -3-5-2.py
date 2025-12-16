
purchases = ["Adidas", "Nike"]
#purchases = ["Nike", "Nike"]
#purchases = []

prices = {'Adidas': 4298, 'Nike': 6550, 'Puma': 4490, 'Asics': 3879}
if len(purchases) == 0:
    print('Ваша корзина пуста')
elif len(purchases) == 2:
        if purchases[0] == purchases[1]:
            discount = 10
            total = prices[purchases[0]]*2
            total_minus_discount = round(total*(1-discount/100),1)
        elif not purchases[0] == purchases[1]:
            discount = 5
            total = prices[purchases[0]]+prices[purchases[1]]
            total_minus_discount = round(total*(1-discount/100),1)
        print(f'Стоимость заказа составила: {total}. С учетом скидки в {discount}% — {total_minus_discount}')
elif len(purchases) == 1:
        total = prices[purchases[0]]
        print(f'Стоимость заказа составила: {total}')
else:
    print('Исходные данные не соответствуют условию задачи')



