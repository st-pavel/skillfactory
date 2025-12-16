## Введите свое решение ниже
def get_most_frequent_word(text):
    def get_words_list(text):
        punctuation_list = ['.', ',', ';', ':', '...', '!', '?', '-', '"', '(', ')']
        text_result = text.lower()
        for punctuation in punctuation_list:
            text_result = text_result.replace(punctuation,'')
        text_result = text_result.split()
        return text_result
    
    def get_unique_words(words_list):
        words_set = set(words_list)
        words_list_result = list(words_set)
        words_list_result.sort()
        return words_list_result
    
    result_text_dict = {}
    result_text = get_words_list(text)
    unique_words_list = get_unique_words(result_text)

    for unique_word in unique_words_list:
        result_text_dict[unique_word] = result_text.count(unique_word)
    
    sorted_result_text_dict = dict(sorted(result_text_dict.items(), key=lambda item: item[1]))

    max_value = 0
    max_sorted_result_word = ''    

    for key in result_text_dict.keys():
        if result_text_dict[key] > max_value:
            max_value = result_text_dict[key]
            max_sorted_result_word = key
    
    #print(max_value)

    for key in result_text_dict.keys():
        if result_text_dict[key] ==  max_value and max_sorted_result_word > key:
            max_sorted_result_word = key 

    return max_sorted_result_word


text_example = "A beginning is the time for taking the most delicate care that the balances are correct. This every sister of the Bene Gesserit knows. To begin your study of the life of Muad'Dib, then take care that you first place him in his time: born in the 57th year of the Padishah Emperor, Shaddam IV. And take the most special care that you locate Muad'Dib in his place: the planet Arrakis. Do not be deceived by the fact that he was born on Caladan and lived his first fifteen years there. Arrakis, the planet known as Dune, is forever his place."
print(get_most_frequent_word(text_example))
# the

text_example = "Есть урок, который идет не сорок пять минут, а всю жизнь. Этот урок проходит и в классе, и в поле, и дома, и в лесу. Я назвал этот урок седьмым потому, что в школе обычно бывает не больше шести уроков. Не удивляйтесь, если я скажу, что учителем на этом уроке может быть и береза возле вашего дома, и бабушка, и вы сами (В. Песков)"
print(get_most_frequent_word(text_example))
# и
