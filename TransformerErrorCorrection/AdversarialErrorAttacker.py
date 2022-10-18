from ErrorCreator import change, get_qwerty
import random


def return_if(sentence, word_pos, corrupt_word, model, y):
    sentence[word_pos] = corrupt_word
    new = model(' '.join(sentence)).squeeze()
    if new[y] < new[1 - y]:
        return ' '.join(sentence)


def find_working_attack(sentence, y, word_pos, model, attack_type):
    sentence_list = sentence.split()
    word = sentence_list[word_pos]

    if attack_type == 'all':
        qwerty = get_qwerty()
        base = model(sentence.unsqueeze(0)).squeeze()
        # assuming binary classification with two output variables
        if base[y] > base[1 - y]:
            attack_modes = [0, 1, 2, 3]
            random.shuffle(attack_modes)
            for i in attack_modes:
                if i == 0:
                    attack_swap(word, i, sentence, word_pos, model, y)
                if i == 1:
                    attack_drop(word, i, sentence, word_pos, model, y)

                if i == 2:
                    attack_add(word, i, sentence, word_pos, model, y)
                if i == 3:
                    attack_replace(word, i, sentence, word_pos, model, y, qwerty)

    if attack_type == 0:
        attack_swap(word, attack_type, sentence, word_pos, model, y)
    elif attack_type == 1:
        attack_drop(word, attack_type, sentence, word_pos, model, y)
    elif attack_type == 2:
        attack_add(word, attack_type, sentence, word_pos, model, y)
    else:
        qwerty = get_qwerty()
        attack_replace(word, attack_type, sentence, word_pos, model, y, qwerty)


def attack_swap(word, attack_type, sentence, word_pos, model, y):
    for char in range(1, len(word) - 1):
        for j in (-1, 1):
            corrupt_word = change(attack_type, char, word, j)
            return_value = return_if(sentence, word_pos, corrupt_word, model, y)
            if return_value:
                return return_value


def attack_drop(word, attack_type, sentence, word_pos, model, y):
    for char in range(1, len(word) - 1):
        corrupt_word = change(attack_type, char, word, None)
        return_value = return_if(sentence, word_pos, corrupt_word, model, y)
        if return_value:
            return return_value


def attack_add(word, attack_type, sentence, word_pos, model, y):
    for char in range(1, len(word) - 1):
        for char2 in '12345657890qwertyuiopasdfghjklzxcvbnm':
            corrupt_word = change(attack_type, char, word, char2)
            return_value = return_if(sentence, word_pos, corrupt_word, model, y)
            if return_value:
                return return_value


def attack_replace(word, attack_type, sentence, word_pos, model, y, qwerty):
    for char in range(1, len(word) - 1):
        for char2 in qwerty[char]:
            corrupt_word = change(attack_type, char, word, char2)
            return_value = return_if(sentence, word_pos, corrupt_word, model, y)
            if return_value:
                return return_value
