## Введите свое решение ниже
def print_personal_data(**personal_data):
    sorted_list = sorted(personal_data.items(), key=lambda x: x[0])
    #print(sorted_list, type(sorted_list))
    result_dict = dict(sorted_list)
    for key, value in result_dict.items():
        print(f'{key}: {value}')
    #    pass 
    return result_dict

print_personal_data(first_name='John', last_name='Doe', age=28, position='Python developer')
# age: 28
# first_name: John
# last_name: Doe
# position: Python developer

print_personal_data(first_name='Jack', last_name='Smith', age=32, work_experience = '5 years', position='Project manager')
# age: 32
# first_name: Jack
# last_name: Smith
# position: Project manager
# work_experience: 5 years