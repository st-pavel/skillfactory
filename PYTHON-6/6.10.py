# не работает в тренировочном редакторе https://skillfactory.itresume.ru/ без изменения лимита рекурсии
import sys
sys.setrecursionlimit(3000)

data = {
    "type": "video",
    "videoID": "vid001",
    "links": [
        {"type":"video", "videoID":"vid002", "links":[]},
        {   "type":"video",
            "videoID":"vid003",
            "links": [
            {"type": "video", "videoID":"vid004"},
            {"type": "video", "videoID":"vid005"},
            ]
        },
        {"type":"video", "videoID":"vid006"},
        {   "type":"video",
            "videoID":"vid007",
            "links": [
            {"type":"video", "videoID":"vid008", "links": [
                {   "type":"video",
                    "videoID":"vid009",
                    "links": [{"type":"video", "videoID":"vid010"}]
                }
            ]}
        ]},
    ]
}

result = [] #Инициализируем результирующий список

def find_video(data, first_recursion_step=True):
    # Создаём цикл по элементам словаря
    global result
    if first_recursion_step: result = [] #Инициализируем результирующий список только при первом запуске рекусии
    if type(data) is dict:
        for key, value in data.items():
            if type(value) is str:
                if key == "videoID":  result.append(value)
            elif type(value) is dict:
                find_video(value, False)
            elif type(value) is list:
                find_video(value, False)
    elif type(data) is list: # проверка, что переданная структура - лист и вызов рекурсии
        for item in data:
            if type(item) is dict:
                find_video(item, False) # внутренний элемент структуры - словарь  ==> вызов рекурсии
            elif type(item) is list:
                find_video(item, False) # внутренний элемент структуры - лист ==> вызов рекурсии
    return result

print(find_video(data))



#Result
#['vid001', 'vid002', 'vid003', 'vid004', 'vid005', 'vid006', 'vid007', 'vid008', 'vid009', 'vid010']


#result = []
#проверка модуля

print(find_video(data = {'type': 'video', 'videoID': 'vid001', 'links': [{'type': 'video', 'videoID': 'vid002', 'links': []}, {'type': 'video', 'videoID': 'vid003', 'links': [{'type': 'video', 'videoID': 'vid004'}, {'type': 'video', 'videoID': 'vid005'}]}, {'type': 'video', 'videoID': 'vid006'}, {'type': 'video', 'videoID': 'vid007', 'links': [{'type': 'video', 'videoID': 'vid008', 'links': [{'type': 'video', 'videoID': 'vid009', 'links': [{'type': 'video', 'videoID': 'vid010'}]}]}]}]}))
#['vid001', 'vid002', 'vid003', 'vid004', 'vid005', 'vid006', 'vid007', 'vid008', 'vid009', 'vid010']

#result = []
print(find_video(data = {'type': 'video', 'links': [{'type': 'video', 'videoID': 'vid001', 'links': [{'type': 'video', 'videoID': 'vid002'}, {'type': 'video', 'videoID': 'vid003'}]}, {'type': 'video', 'videoID': 'vid004', 'links': [{'type': 'video', 'videoID': 'vid005', 'links': [{'type': 'video', 'videoID': 'vid006', 'links': [{'type': 'video', 'videoID': 'vid07'}]}]}]}]}))
# ожидаемый ответ
# ['vid001', 'vid002', 'vid003', 'vid004', 'vid005', 'vid006', 'vid07']

