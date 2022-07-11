import os
from glob import glob
import builtins
from unittest.mock import patch
from code_function import preprocess_script, make_dataset
import sys
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
    if num == "problem002":
        code = preprocess_script(problem)
        code = eval2input(code)
        check_std = passstdin(code)
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
                temp = sys.stdout
                sys.stdout = open(f'D:/open/outputs/{name}.txt', 'w')
                try:
                    exec(code)
                except Exception as e:
                    print(e)
                    # if e.args == ('eval() arg 1 must be a string, bytes or code object',):
                    #     # print(1)
                    #     # print(code)
                    # elif e.args == ("'list' object is not callable",):
                    #     # print(2)
                    #     # print(code)
                    # else:
                    #     print(e)
                sys.stdout.close()
                sys.stdout = temp


