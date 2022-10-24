import random
import os
import json


def create_qwerty_adjacents():
    qwerty = {
        'q': 'wsa',
        'a': 'qwsxz',
        'z': 'asx',
        'w': 'qeasd',
        's': 'qweadzxc',
        'x': 'zasdc',
        'e': 'wrsdf',
        'd': 'wersfxcv',
        'c': 'xsdfv',
        'r': 'etdfg',
        'f': 'ertdgcvb',
        'v': 'dfgcb',
        't': 'ryfgh',
        'g': 'rtyfhvbn',
        'b': 'vfghn',
        'y': 'tughj',
        'h': 'tyugjbnm',
        'n': 'bghjm',
        'u': 'hyik',
        'j': 'yuihknm',
        'm': 'nhjk',
        'i': 'uojkl',
        'k': 'uiojlm',
        'o': 'ipkl',
        'l': 'lkiop',
        'p': 'pol',
    }
    key_list = list(qwerty.keys())
    for key in key_list:
        if key.isalpha():
            qwerty[key.upper()] = qwerty[key].upper()
    with open('QWERTY.json', 'w') as f:
        json.dump(qwerty, f)

    return qwerty


def change(type_change, char_to_change, word, choice):
    qwerty = get_qwerty()
    if type_change == 0:
        # Swap adjacent characters
        if char_to_change == 1:
            next_char = 1
        elif char_to_change == len(word) - 2:
            next_char = -1
        else:
            if not choice:
                next_char = random.choice([-1, 1])
            else:
                next_char = choice

        corrupt_word = word[: char_to_change] + word[char_to_change + next_char] + word[char_to_change + 1:]
        corrupt_word = corrupt_word[: char_to_change + next_char] + word[
            char_to_change] + corrupt_word[char_to_change + next_char + 1:]
    elif type_change == 1:
        # Drop a character
        corrupt_word = word[: char_to_change] + word[char_to_change + 1:]
    elif type_change == 2:
        # Add a character
        if not choice:
            add_char = random.choice('qwertyuiopasdfghjklzxcvbnm')
        else:
            add_char = choice

        corrupt_word = word[:char_to_change] + add_char + word[
                                                          char_to_change:]
    else:
        # substitute with an adjacent character
        if not choice:
            try:
                replace = random.choice(qwerty[word[char_to_change]])
            except KeyError:

                replace = 'g'
        else:
            replace = choice
        corrupt_word = word[:char_to_change] + replace + word[char_to_change + 1:]
    return corrupt_word


def random_change(word: str, type_change: int = None, num_changes: int = 1):
    if len(word) < 4:
        return word
    char_to_change = random.randint(1, len(word) - 2)

    if not type_change:
        type_change = random.randint(0, 4)

    corrupt_word = change(type_change, char_to_change, word, None)
    if num_changes == 2:
        if len(corrupt_word) < 4:
            return corrupt_word
        char_to_change = random.randint(1, len(corrupt_word) - 2)

        if not type_change:
            type_change = random.randint(0, 4)
        corrupt_word = change(type_change, char_to_change, corrupt_word, None)
    return corrupt_word


def corrupt_sentence(sentence, prob, type_change: int = None, num_changes: int = 1):
    if random.random() < prob:
        return ' '.join(random_change(word, type_change, num_changes) for word in sentence.split(' ')), 1
    else:
        return sentence, 0


def get_qwerty():
    if "QWERTY.json" in os.listdir():
        with open("QWERTY.json", 'r') as f:
            qwerty = json.load(f)
    else:
        qwerty = create_qwerty_adjacents()
    return qwerty
