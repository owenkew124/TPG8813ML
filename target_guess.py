from transformers import (
    GPT2LMHeadModel
)
import time
import datetime
import threading
import torch
from PagPassGPT.tokenizer import CharTokenizer
import argparse
from itertools import combinations_with_replacement, permutations, product
import random
import re
import logging
import os
import string
from zxcvbn import zxcvbn
from Levenshtein import distance, ratio

MAX_LEN = 32
logging.basicConfig(filename='./generate_pw_variant.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

# We are generating password variants that comply to the minimum common modern password policy:
# 1 uppercase letter, 1 lowercase letter, 1 digit, 1 symbol pattern (u1-l1-d1-s1)

# PagPassGPT simplify tokenization by being case insensitive.
# So we have to randomly upcase a letter from the generated password variant
# and ensure that the password variant pattern is L-S-N
# with minimum length set to 8

def generate_password_pattern(permutation, pw_len):
    parts = permutation.split()
    counts = {part[0]: int(part[1:]) for part in parts}
    max_len = max(8, pw_len)

    for letter in 'LNS':
        if letter not in counts:
            counts[letter] = 0

    total = sum(counts.values())
    while sum(counts.values()) != max_len or min(counts.values()) == 0:
        if sum(counts.values()) < max_len:
            # Add to missing or smallest count
            candidates = [l for l in 'LNS' if counts[l] == min(counts.values())]
            letter = random.choice(candidates)
            counts[letter] += 1
        else:
            # Subtract from largest count
            candidates = [l for l in 'LNS' if counts[l] == max(counts.values()) and counts[l] > 1]
            if candidates:
                letter = random.choice(candidates)
                counts[letter] -= 1
            else:
                # If we can't subtract, redistribute
                for l in 'LNS':
                    if counts[l] > 1:
                        counts[l] -= 1
                        break

    parts = [f"{letter}{count}" for letter, count in counts.items() if count > 0]
    all_permutations = list(permutations(parts))

    permutation_strings = [' '.join(perm) for perm in all_permutations]
    return permutation_strings

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

def remove_random_character(s):
    if not s:
        return s
    index = random.randint(0, len(s) - 1)  # Choose a random index
    return s[:index] + s[index + 1:]

def ensure_case_diversity(password):
    if has_both_cases(password):
        return password

    password_chars = list(password)

    if not has_uppercase(password):
        add_uppercase(password_chars)

    if not has_lowercase(password):
        add_lowercase(password_chars)

    return ''.join(password_chars)

def has_both_cases(string):
    return has_uppercase(string) and has_lowercase(string)

def has_uppercase(string):
    return any(char.isupper() for char in string)

def has_lowercase(string):
    return any(char.islower() for char in string)

def add_uppercase(chars):
    change_random_char(chars, str.islower, str.upper)

def add_lowercase(chars):
    change_random_char(chars, str.isupper, str.lower)

def change_random_char(chars, condition, transform):
    allowed_chars = string.ascii_letters
    eligible_indices = [i for i, char in enumerate(chars) if condition(char)]
    if eligible_indices:
        index = random.choice(eligible_indices)
        transformed_char = transform(chars[index])
        # Ensure the transformed character is in the allowed set
        while transformed_char not in allowed_chars:
            transformed_char = transform(random.choice(list(allowed_chars)))
        chars[index] = transformed_char

def valid_format(password):
    pattern = r'^(?=.*[!@#$%^&*()_+\-=[\]{}|;:\'",.<>/?`~])(?=.*\d).+$'
    return bool(re.match(pattern, password))

def compute_log_likelihood(model, input_ids):
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)  # The labels parameter causes the model to return the loss
        log_likelihood = outputs.loss.item()  # The loss is the negative log likelihood of the sequence
    return log_likelihood

def get_log_filename(input_password):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_password = ''.join(c if c.isalnum() else '_' for c in input_password[:10])
    return f"password_variant_{safe_password}_{timestamp}.log"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="directory of pagpassgpt", type=str, default='./model/checkpoint-384000')
    parser.add_argument("--vocabfile_path", help="path of vocab file", type=str, default='./tokenizer/vocab.json')
    parser.add_argument("--input_password", help="password to be used for creating variants", type=str, required=True)
    parser.add_argument("--generate_num", help="query budget per pattern", default=10, type=int)
    parser.add_argument("--compute_loglikelihood", help="compute log likelihood of the passwords", action="store_true")
    args = parser.parse_args()

    model_path = args.model_path
    vocab_file = args.vocabfile_path
    pw = args.input_password
    orig_pw= args.input_password

    logger = logging.getLogger()
    logger.info(f"Script started with arguments: {vars(args)}")

    logger.info(f'Load tokenizer.')
    tokenizer = CharTokenizer(vocab_file=vocab_file,
                              bos_token="<BOS>",
                              eos_token="<EOS>",
                              sep_token="<SEP>",
                              unk_token="<UNK>",
                              pad_token="<PAD>"
                              )
    tokenizer.padding_side = "left"
    logger.info(f'Loaded: {tokenizer.encoder}')

    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    inputs = set()

    ip = ' '.join(get_pattern(pw))
    fps = generate_password_pattern(ip, len(pw))
    logger.info(f"{pw} patterns: {fps}")
    for fp in fps:
        inputs.add(fp + ' <SEP> ' + ' '.join(list(pw)))

    # simulate Das-R rule: Delete Character
    pw = remove_random_character(args.input_password)
    ip = ' '.join(get_pattern(pw))
    fps = generate_password_pattern(ip, len(pw))
    logger.info(f"{pw} patterns: {fps}")
    for fp in fps:
        inputs.add(fp + ' <SEP> ' + ' '.join(list(pw)))

    pw = remove_random_character(pw)
    ip = ' '.join(get_pattern(pw))
    fps = generate_password_pattern(ip, len(pw))
    logger.info(f"{pw} patterns: {fps}")
    for fp in fps:
        inputs.add(fp + ' <SEP> ' + ' '.join(list(pw)))
    logger.info(f"input tokens: {inputs}")
    tokenizer_forgen_results = [tokenizer.encode_forgen(input_text) for input_text in inputs]
    passwords = set()

    for tokenizer_forgen_result in tokenizer_forgen_results:
        start_time = time.time()
        input_ids=tokenizer_forgen_result.view([1, -1])
        outputs = model.generate(
            input_ids=input_ids,
            pad_token_id=tokenizer.pad_token_id,
            max_length=MAX_LEN,
            do_sample=True,
            num_return_sequences=args.generate_num
        )
        end_time = time.time()
        logger.info(f'-- generation time: {(end_time - start_time):.6f} secs')
        decoded_outputs = tokenizer.batch_decode(outputs)
        for output in decoded_outputs:
            pattern, pw_variant = output.split(' ', 1)
            if valid_format(pw_variant) and len(pw_variant) >= 8 and orig_pw is not pw_variant:
                password = ensure_case_diversity(pw_variant)
                passwords.add(password)

    num = args.generate_num
    if num > len(passwords):
        num = len(passwords)
    logger.info(f'====selecting {num} from {len(passwords)} password variants')
    selected_passwords = random.sample(list(passwords), num)
    inputs = set()
    for pw_variant in sorted(selected_passwords):
        fp = ' '.join(get_pattern(pw_variant))
        inputs.add(fp + ' <SEP> ' + ' '.join(list(pw_variant)))
        print(pw_variant)

    if args.compute_loglikelihood:
        orig_zxcvbn = zxcvbn(orig_pw)
        orig_pattern = ''.join(get_pattern(orig_pw))
        logger.info(f'selected input tokens: {inputs}')
        tokenizer_forgen_results = [tokenizer.encode_forgen(input_text) for input_text in inputs]
        for tokenizer_forgen_result in tokenizer_forgen_results:
            input_ids=tokenizer_forgen_result.view([1, -1])
            log_likelihood = compute_log_likelihood(model, input_ids)
            pattern, pw_variant = tokenizer.decode(tokenizer_forgen_result).split(' ', 1)
            pw_zxcvbn = zxcvbn(pw_variant)
            edit_distance = distance(orig_pw, pw_variant)
            similarity_ratio = ratio(orig_pw, pw_variant)
