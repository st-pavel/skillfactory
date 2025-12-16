import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell
import os

nb_path = '/home/pavel/IDE/skillfactory/DS_PROD-4/DS_PROD-4_001.ipynb'
if not os.path.exists(nb_path):
    print(f"Notebook not found at {nb_path}")
    exit(1)

nb = nbformat.read(nb_path, as_version=4)

# Markdown cell
md_source = "### Проверка на нормальность (Тест Шапиро-Уилка)\n\nПроверим нормальность распределения ежедневной конверсии в группах с помощью теста Шапиро-Уилка.\n\nНулевая гипотеза ($H_0$): Распределение данных является нормальным.\nАльтернативная гипотеза ($H_1$): Распределение данных отлично от нормального.\n\nУровень значимости $\\alpha = 0.05$."
nb.cells.append(new_markdown_cell(md_source))

# Code cell
code_source = """from scipy.stats import shapiro

# Выбираем ежедневную конверсию для группы А и B
daily_conversion_a = daily_data[daily_data['group'] == 'A']['conversion']
daily_conversion_b = daily_data[daily_data['group'] == 'B']['conversion']

# Проводим тест Шапиро-Уилка
shapiro_result_a = shapiro(daily_conversion_a)
shapiro_result_b = shapiro(daily_conversion_b)

print(f"Группа A: p-value = {shapiro_result_a.pvalue:.4f}")
print(f"Группа B: p-value = {shapiro_result_b.pvalue:.4f}")

# Интерпретация
alpha = 0.05
if shapiro_result_a.pvalue < alpha or shapiro_result_b.pvalue < alpha:
    print("Распределение конверсии в одной или обеих группах отлично от нормального.")
else:
    print("Распределение конверсии в обеих группах нормальное.")
"""
nb.cells.append(new_code_cell(code_source))

nbformat.write(nb, nb_path)
print("Notebook updated.")
