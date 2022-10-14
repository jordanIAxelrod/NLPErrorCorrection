import random
import os
import json

if "QWERTY.json" in os.listdir():
    with open("QWERTY.json", r) as f:
        qwerty = json.load(f)
else:
    qwerty = create_qwerty_adjacents()


def create_qwerty_adjacents():
    qwerty = {
        'q': '12wsa',
        'a': 'qwsxz',
        'z': 'asx',
        'w': '123qeasd',
        's': 'qweadzxc',
        'x': 'zasdc',
        'e': '234wrsdf',
        'd': 'wersfxcv',
        'c': 'xsdfv',
        'r': '345etdfg',
        'f': 'ertdgcvb',
        'v': 'dfgcb',
        't': '456ryfgh',
        'g': 'rtyfhvbn',
        'b': 'vfghn',
        'y': '567tughj',
        'h': 'tyugjbnm',
        'j': 'yuihknm',
        'm': 'nhjk',
        'i': '789uojkl',
        'k': 'uiojlm',
        'o': '890ipkl',
        'l': 'lkiop',
        'p': '09pol'
    }
    with open('QWERTY.json', 'w') as f:
        json.dump(qwerty, f)

    return qwerty


def random_change(word: str, type_change: int = None, num_changes: int = 1):
    if len(word) < 4:
        return word
    char_to_change = random.randint(1, len(word) - 1)

    if not type_change:
        type_change = random.randint(0, 4)

    if type_change == 0:
        # Swap adjacent characters
        if char_to_change == 1:
            next_char = 1
        elif char_to_change == len(word) - 2:
            next_char = -1
        else:
            next_char = random.choice([-1, 1])
        corrupt_word = word[: char_to_change] + word[char_to_change + next_char] + word[char_to_change + 1:]
        corrupt_word = corrupt_word[: char_to_change + next_char] + word[
            char_to_change] + corrupt_word[char_to_change + next_char + 1:]
    elif type_change == 1:
        # Drop a character
        corrupt_word = word[: char_to_change] + word[char_to_change + 1:]
    elif type_change == 2:
        # Add a character
        corrupt_word = word[:char_to_change] + random.choice('12345657890qwertyuiopasdfghjklzxcvbnm') + word[
                                                                                                        char_to_change:]
    else:
        # substitute with an adjacent character
        replace = qwerty[word[char_to_change]]
        corrupt_word = word[:char_to_change] + random.choice(replace) + word[char_to_change + 1:]
    return corrupt_word


def corrupt_sentence(sentence, type_change: int = None, num_changes: int = 1):
    return ' '.join(random_change(word, type_change, num_changes) for word in sentence.split(' '))

