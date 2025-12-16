## Введите свое решение ниже
def get_words_list(text):
    punctuation_list = ['.', ',', ';', ':', '...', '!', '?', '-', '"', '(', ')']
    text_result = text.lower()
    for punctuation in punctuation_list:
        text_result = text_result.replace(punctuation,'')
    text_result = text_result.split()
    return text_result

text_example = "Arrakis, the planet known as Dune, is forever his place."

print(get_words_list(text=text_example))
# ['arrakis', 'the', 'planet', 'known', 'as', 'dune', 'is', 'forever', 'his', 'place']
