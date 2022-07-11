import os
from glob import glob
import builtins
from unittest.mock import patch
from code_function import preprocess_script, make_dataset
import sys
import subprocess

def eval2input(code):
    code = code.replace("eval(input())", "input()")
    return code

def passstdin(code):
    if ("stdin" in code) or ("fileinput.input()" in code):
        return 1
    else:
        return 0

problems= sorted(glob('D:/open/code/*/*.py'))
for problem in problems:
    name = os.path.basename(problem)
    num = name.split("_")[0]
    code = preprocess_script(problem)
    code = eval2input(code)
    check_std = passstdin(code)
    # print(name)
    if check_std == 1:
        continue
    else:
        with patch('builtins.input') as input_mock:
            input_mock.side_effect = [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                0
            ]
            # sys.stdout = open(f'D:/open/outputs/{name}.txt', 'w')
            try:
                p = subprocess.run(['python', problem])
            except Exception as e:
                print(e)
# sys.stdout.close()
