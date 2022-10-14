from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm


# 函数参数：seq：氨基酸序列； maxK：最大间隔数
def seq_to_KSAAP_code(seq, maxK):
    total_num = 0
    # 按英文字母表顺序排序的21种氨基酸字母序列
    alphabet = 'ARNDCQEGHILKMFPSTWYVX'
    vec = []
    sequence = seq

    # k=0,1,2,3,4的情况
    for k in range(maxK + 1):
        # k=n 时，子串的总数
        total_num = len(sequence) - k - 1
        # k变化时，subSeq_count、subSeq_frequency 计数清零（累加则定义放for循环外）
        subSeq_count = defaultdict(int)
        subSeq_frequency = defaultdict(int)

        # k=n 时，计算子序列出现的次数
        for i in range(total_num):
            sub_seq = sequence[i:i + k + 2:k + 1]
            subSeq_count[sub_seq] += 1

        # k=n 时，计算子序列出现的频率 subSeq_frequency
        for sub_s, count in subSeq_count.items():
            # print(sub_s,count)
            subSeq_frequency[sub_s] = float(subSeq_count[sub_s]) / total_num
        #         print(subSeq_count, subSeq_frequency)

        # 按字母表顺序组合氨基酸对，并查找此氨基酸对出现的频率，组成向量vec
        for s1 in alphabet:
            for s2 in alphabet:
                sub_alp = s1 + s2
                vec.append(subSeq_frequency[sub_alp])

    vec_arr = np.array(vec)
    vec_arr = vec_arr.reshape(k + 1, len(alphabet) ** 2)
    #     print(vec_arr)
    #     print(vec, len(vec))
    return vec


"""
# data为dataframe格式，且序列列名为“seq”
"""


def get_features(data, maxK):
    if isinstance(data, pd.DataFrame):
        sequences = data.seq
    else:
        sequences = data
    df_features = pd.DataFrame()
    for seq in tqdm(sequences):
        fea = seq_to_KSAAP_code(seq, maxK)
        df_features = df_features.append(pd.DataFrame([fea]), ignore_index=True)
    return df_features

# test_seq_1 = 'YVHVNATYVNVKCVAPYPSLLSSEDNADDEVDT'
# test_seq_2 = 'AAADDCCCAAAKCVAPYPSLLRTRDNADDEVDT'
# # test_seq = 'YVHVYV'
# fea = seq_to_KSAAP_code(test_seq_1, 4)
# print(len(fea), k_code)
# # # print(sum(k_code[400:800]))
# # df = pd.DataFrame([k_code])
# data = pd.DataFrame([[test_seq_1], [test_seq_2]],columns=['seq'])
# df = get_features(data, 4)
# df

# test
# seq_1 = 'AVGIGTVHQQQHEDILSKTFTQXXXXXXXXXXXXX'
# seq_2 = 'YVHVNATYVNVKCVAPYPSLLSSEDNADDEVDTSS'
# seq_3 = 'AAADDCCCAAAKCVAPYPSLLRTRDNADDEVDTSS'
#
# df_data = pd.DataFrame([seq_1, seq_2, seq_3], columns=['seq'])
# df_data['label'] = 1
# fea = get_features(df_data, 4)
# print(fea.shape, fea)
# np_arr = np.array(fea)
# np_fea = np.reshape(np_arr, (len(df_data), -1, 21))
# print(np_fea.shape, np_fea)
