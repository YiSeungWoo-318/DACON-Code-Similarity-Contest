from tqdm import tqdm
import pandas as pd
from rank_bm25 import BM25Okapi
from itertools import combinations

def preprocess_script(script):
    with open(script,'r',encoding='utf-8') as file:
        lines = file.readlines()
        preproc_lines = []
        for line in lines:
            line = line.replace("eval(input())", "input()")
            if line.lstrip().startswith('#'):
                continue
            line = line.rstrip()
            if '#' in line:
                line = line[:line.index('#')]
            line = line.replace('\n','')
            line = line.replace('    ','\t')
            if line == '':
                continue
            preproc_lines.append(line)
    return preproc_lines

def indentation(text):
    re_text = []
    #unified
    for t in text:
        t = t.replace("\t","        ") #8
        ls = t.lstrip()
        remain = len(t) - len(ls)
        if remain:
            if remain % 4 != 0:
                t = "  " + t
        else:
            pass

    #convert \t
        ls = t.lstrip()
        remain = len(t) - len(ls)
        if remain:
            start = t[:remain]
            start = start.replace("    ","\t")
            t = start + t[remain:]
        else:
            pass

        re_text.append(t)
    data_script = "\n".join(re_text)
    return data_script

def make_dataset(train_df, tokenizer):
    # import random
    codes = train_df['code'].to_list()
    problems = train_df['problem_num'].unique().tolist()
    problems.sort()

    tokenized_corpus = [tokenizer.tokenize(code) for code in codes]
    bm25 = BM25Okapi(tokenized_corpus)

    total_positive_pairs = []
    total_negative_pairs = []

    for problem in tqdm(problems):
        solution_codes = train_df[train_df['problem_num'] == problem]['code']
        positive_pairs = list(combinations(solution_codes.to_list(),2))
        #reduce positive_pairs
        # random.seed(1)
        # random.shuffle(positive_pairs)
        # len(positive_pairs)

        solution_codes_indices = solution_codes.index.to_list()
        negative_pairs = []

        first_tokenized_code = tokenizer.tokenize(positive_pairs[0][0])
        negative_code_scores = bm25.get_scores(first_tokenized_code)
        negative_code_ranking = negative_code_scores.argsort()[::-1] # 내림차순
        ranking_idx = 0

        for solution_code in solution_codes:
            negative_solutions = []
            while len(negative_solutions) < len(positive_pairs) // len(solution_codes):
                high_score_idx = negative_code_ranking[ranking_idx]

                if high_score_idx not in solution_codes_indices:
                    negative_solutions.append(train_df['code'].iloc[high_score_idx])
                ranking_idx += 1

            for negative_solution in negative_solutions:
                negative_pairs.append((solution_code, negative_solution))

        total_positive_pairs.extend(positive_pairs)
        total_negative_pairs.extend(negative_pairs)

    pos_code1 = list(map(lambda x:x[0],total_positive_pairs))
    pos_code2 = list(map(lambda x:x[1],total_positive_pairs))

    neg_code1 = list(map(lambda x:x[0],total_negative_pairs))
    neg_code2 = list(map(lambda x:x[1],total_negative_pairs))

    pos_label = [1]*len(pos_code1)
    neg_label = [0]*len(neg_code1)

    pos_code1.extend(neg_code1)
    total_code1 = pos_code1
    pos_code2.extend(neg_code2)
    total_code2 = pos_code2
    pos_label.extend(neg_label)
    total_label = pos_label
    pair_data = pd.DataFrame(data={
        'code1':total_code1,
        'code2':total_code2,
        'similar':total_label})
    pair_data = pair_data.sample(frac=1).reset_index(drop=True)
    return pair_data

def reduction_dataset(data):
    data1 = data.drop_duplicates("code1")
    data2 = data.drop_duplicates("code2")
    re_data = pd.concat([data1, data2], ignore_index=True)
    re_data = re_data.reset_index(drop=True)
    return re_data

def reduction_xdataset(data):
    data1 = data.drop_duplicates("code1")
    data2 = data.drop_duplicates("code2")
    data3 = data.drop_duplicates("code1", keep="last")
    data4 = data.drop_duplicates("code2", keep="last")
    re_data = pd.concat([data1, data2, data3, data4], ignore_index=True)
    re_data = data.drop_duplicates(["code1", "code2"])
    re_data = re_data.reset_index(drop=True)
    return re_data

