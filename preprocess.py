import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from rapidfuzz.distance import Levenshtein
from typing import Dict, Tuple, List
from tqdm.auto import tqdm
from pathlib import Path

data_path = Path(r'C:\Users\david\PhD\Class\TPG8813\51k_training_set.txt')
processed_data_path = Path(r'C:\Users\david\PhD\Class\TPG8813\51k_training_set_processed.txt')


user_passwords = {}
num_no_colon_lines = 0
num_higher_than_length_lim = 0
process_to_pag_pass_GPT = True

max_length = 20
if __name__ == '__main__':
    with open(data_path, "r") as f:
        for i, line in tqdm(enumerate(f), desc="Reading in data"):
            try:
                # strip newline from end of line
                line = line.strip('\n')  # only remove newline since passwords can have leading/trailing spaces

                # split line into user and password
                if ":" not in line:
                    num_no_colon_lines += 1
                    continue

                user, password = line.split(":", 1)  # only split on first colon to allow for colons in password

                if len(password) > max_length:
                    num_higher_than_length_lim+=1
                    continue

                if user not in user_passwords:
                    user_passwords[user] = []


                user_passwords[user].append(password)
            except ValueError as e:
                print(f"Error on line {i}: {line}")
                print(f"Error: {e}")
                break
    print(f"User Count Number : {len(user_passwords.keys())} \n")
    print(f"Longest Password : {max_length} \n")
    print(f"Number of invalid formatted user:password lines : {num_no_colon_lines} \n")
    print(f"Number higher than max length : {num_higher_than_length_lim} \n")


    num_users_filtered = 0
    ''' How many passwords should we let the user have before theyre used in the training?'''
    pass_threshold_count = 2
    ''' 
        For now remove the users without multiple passwords. 
        Can change this to duplicate an existing password if too few passwords.
    '''
    for user in user_passwords:
        if len(user_passwords[user]) < pass_threshold_count:
            user_passwords[user].pop()
            num_users_filtered += 1
    print(f"User Count Number After Filter: {len(user_passwords.keys())} \n")
    print(f"Users filtered: {num_users_filtered} \n")
    if process_to_pag_pass_GPT:
        f_out = open(processed_data_path, 'w', encoding='utf-8', errors='ignore')
        for user in user_passwords:
            for password in user_passwords[user]:
                f_out.write(password)
                f_out.write("\n")
        f_out.close()

    ''' DATASET SPLITTING INTO TRAIN TEST'''
    import random
    def split_train_test(file_path, ratio, train_path, test_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        print('Shuffling passwords.')
        random.shuffle(lines)
        f.close()

        split = int(len(lines) * ratio)

        with open(train_path, 'w') as f:
            print('Saving 80% ({}) of dataset for training in {}'.format(split, train_path))
            f.write(''.join(lines[0:split]))
        f.close()

        with open(test_path, 'w') as f:
            print('Saving 20% ({}) of dataset for test in {}'.format(len(lines) - split, test_path))
            f.write(''.join(lines[split:]))
        f.close()


    train_path = Path(r"C:\Users\david\PhD\Class\TPG8813\51k_training_set_processed_train.txt")
    test_path = Path(r"C:\Users\david\PhD\Class\TPG8813\51k_training_set_processed_test.txt")
    ratio = 0.8
    print(f'Split begin...\n')
    split_train_test(processed_data_path, ratio, train_path, test_path)
    print(f'Split done...\n')


    def get_pattern(password: str):
        result = []

        current_type = None
        current_length = 0

        for char in password:
            if char.isalpha():
                if current_type == 'L':
                    current_length += 1
                else:
                    if current_type:
                        result.append(current_type + str(current_length))
                    current_type = 'L'
                    current_length = 1
            elif char.isdigit():
                if current_type == 'N':
                    current_length += 1
                else:
                    if current_type:
                        result.append(current_type + str(current_length))
                    current_type = 'N'
                    current_length = 1
            else:
                if current_type == 'S':
                    current_length += 1
                else:
                    if current_type:
                        result.append(current_type + str(current_length))
                    current_type = 'S'
                    current_length = 1

        if current_type:
            result.append(current_type + str(current_length))
        return result


    complete_training_dataset = Path(r"C:\Users\david\PhD\Class\TPG8813\51k_training_set_processed_train_COMPLETE.txt")
    training_data_set_file = open(train_path, 'r', encoding='utf-8', errors='ignore')

    f_out = open(complete_training_dataset, 'w', encoding='utf-8', errors='ignore')
    lines = training_data_set_file.readlines()

    vocab = r"C:\Users\david\PhD\Class\TPG8813\PagPassGPT\tokenizer\vocab.json"
    import json

    with open(vocab, 'r') as file:
        vocab_dict = json.load(file)

    num_filtered = 0
    for line in lines:
        write_line = True
        password = line[:-1]
        prompt = ' '.join(get_pattern(password))
        new_line = prompt + ' <SEP> ' + ' '.join(list(password)) + '\n'
        new_line_check = new_line[0:-1]
        tokens = new_line.strip("\n").split(" ")
        for token in tokens:
            if token not in vocab_dict.keys():
                num_filtered += 1
                write_line = False
                break
        if write_line == True:
            f_out.write(new_line)
    f.close()
    print(f"Number filtered that werent in the vocab: {num_filtered} \n")

    PCFG_rate_file = Path(r"C:\Users\david\PhD\Class\TPG8813\PCFG.txt")

    f_in = open(train_path, 'r', encoding='utf-8', errors='ignore')
    f_out = open(PCFG_rate_file, 'w', encoding='utf-8', errors='ignore')

    pcfg_patterns_dict = {}

    lines = f_in.readlines()
    total_num = len(lines)
    for line in lines:
        if not line:
            continue
        password = line[:-1]
        pcfg_pattern = ' '.join(get_pattern(password))
        if pcfg_pattern in pcfg_patterns_dict:
            pcfg_patterns_dict[pcfg_pattern] += 1
        else:
            pcfg_patterns_dict[pcfg_pattern] = 1

    pcfg_patterns_dict = dict(sorted(pcfg_patterns_dict.items(), key=lambda x: x[1], reverse=True))

    for key, value in pcfg_patterns_dict.items():
        f_out.write(f'{key}\t{value / total_num}\n')