import os
from subprocess import check_output, CalledProcessError
import time

def run_guessing(user, password):
    model_path = r"PagPassGPT/model/checkpoint-6000"
    vocabfile_path = r"PagPassGPT/tokenizer/vocab.json"
    script_path = r"C:\Users\david\PhD\Class\TPG8813\target_guess.py"
    run_cmd = f"python {script_path} --model_path {model_path} --vocabfile_path {vocabfile_path} --user {user} --input_password {password}"
    try:
        out = check_output(run_cmd, shell=True).decode("utf-8")
        t = (0, out)
    except CalledProcessError as e:
        t = (e.returncode, e.output)
    return t

if __name__ == '__main__':
    from pathlib import Path
    data_path = Path(r"C:\Users\david\PhD\Class\TPG8813\51k_training_set.txt")
    user_passwords = {}
    max_length = 12
    ''' Create dict. For all users'''
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            # strip newline from end of line
            line = line.strip('\n')  # only remove newline since passwords can have leading/trailing spaces

            # split line into user and password
            if ":" not in line:
                continue




            user, password = line.split(":", 1)  # only split on first colon to allow for colons in password

            if len(password) > max_length:
                continue

            if user not in user_passwords:
                user_passwords[user] = []

            user_passwords[user].append(password)
    passwords_path = r"C:\Users\david\PhD\Class\TPG8813\generated_passwords.txt"
    passwords_skipped = 0
    with open(passwords_path, "a") as f:
        for user in user_passwords.keys():

            if len(user_passwords[user]) > 2:
                passwords = user_passwords[user][0:2]
            else:
                passwords = user_passwords[user]

            for password in passwords:
                start_time = time.time()
                return_code, output = run_guessing(user, password)
                if return_code == 0:
                    output = output[0:-4]
                    f.write(output)
                    f.flush()
                else:
                    passwords_skipped += 1
                    print(f" Exception for user: {user} and password: {password}\n")
                    continue
                print(f"Elapsed Time for Generation and write to file:{time.time()-start_time}")

    print(f"Number of passwords skipped: {passwords_skipped}\n")
