import re
from collections import Counter
import pandas as pd
from tqdm import tqdm
import numpy as np
"""
# 函数说明：获取序列EAAC编码
    函数参数：
        df_data: DataFrame格式，必须有序列seq列
        window：滑动窗口长度
        unknown_acid：未知氨基酸表示
"""
def get_EAAC_features(df_data, window=5, unknown_acid='X'):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    AA += unknown_acid

    df_features = pd.DataFrame()

    for row in tqdm(range(len(df_data))):
        if isinstance(df_data, pd.DataFrame):
            sequence = df_data.loc[row].seq
        else:
            sequence = df_data[row]
        list_fea = []
        for j in range(len(sequence)):
            if j < len(sequence) and j + window <= len(sequence):
                # 去除unknown_acid
                # count = Counter(re.sub(unknown_acid, '', sequence[j:j+window]))
                # 不去除unknown_acid
                count = Counter(sequence[j:j + window])
                for key in count:
                    # 计算频率（去除unknown_acid）
                    # count[key] = count[key] / len(re.sub(unknown_acid, '', sequence[j:j+window]))
                    # 计算频率（不去除unknown_acid）
                    count[key] = count[key] / len(sequence[j:j + window])
                for aa in AA:
                    list_fea.append(count[aa])
        df_features = df_features.append(pd.DataFrame([list_fea]), ignore_index=True)
    return df_features

# test
# seq_1 = 'AVGIGTVHQQQHEDILSKTFTQXXXXXXXXXXXXX'
# seq_2 = 'YVHVNATYVNVKCVAPYPSLLSSEDNADDEVDTSS'
# seq_3 = 'AAADDCCCAAAKCVAPYPSLLRTRDNADDEVDTSS'
# #
# df_data = pd.DataFrame([seq_1, seq_2, seq_3], columns=['seq'])
# df_data['label'] = 1
# fea = get_EAAC_features(df_data, 5)
# print(fea.shape, fea)
#
# import fea_one_hot
# fea1 = fea_one_hot.get_onehot_features(df_data)
# print(fea1.shape, fea1)
# np_arr = np.array(fea)
# np_fea = np.reshape(np_arr, (len(df_data), -1, 21))
# print(np_fea.shape, np_fea)
