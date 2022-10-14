from IPython.core.display import display
import os
import pandas as pd

"""
# 函数说明：
    将数据集格式化为以下形式：
    df_data为DataFrame格式：
    seq    |   label
    xxxxx  |    1
           .
           .
           .
    xxxxx  |     0
    其中label为1表示正样本，label为0表示负样本
"""


def format_data(df_source_data, label):
    df_data = pd.DataFrame(df_source_data.Peptide)
    df_data.columns = ['seq']
    df_data['label'] = label
    return df_data


"""
# 函数说明：
    获取不符合氨基酸字母范围的错误序列的索引
    # 参数说明：
    sr_source_seq：series格式，氨基酸序列
    unknown_acid：str格式，未知氨基酸的字母表示
"""


def get_worng_acid_seq_index(sr_source_seq, str_unknown_acid):
    str_wrong_acid = 'BJOUXZ'
    list_wrong_acid = list(str_wrong_acid)
    try:
        list_wrong_acid.remove(str_unknown_acid)
    except ValueError:
        print("str_unknown_acid{} is not in the list".format(str_unknown_acid))

    list_wrong_seq_index = []
    for i in range(len(sr_source_seq)):
        # find()返回查找字符的索引
        list_result = [str(sr_source_seq[i]).find(j) for j in list_wrong_acid]
        if list_result.count(-1) != len(list_result):
            #             print(list_result)
            list_wrong_seq_index.append(i)
    return list_wrong_seq_index


"""
# 函数说明：
    删除不合法序列
    # 参数说明：
    df_data：DataFrame格式，原始数据集
    str_data：str格式，数据集来源：如train_pos_data，便于区别打印
"""


def delete_wrong_data(df_data, str_data):
    list_wrong_seq_index = get_worng_acid_seq_index(df_data.Peptide, 'X')
    print("wrong_data in {}:".format(str_data))
    display(df_data.loc[list_wrong_seq_index])
    df_proper_data = df_data.drop(index=list_wrong_seq_index)
    return df_proper_data


def get_processed_data(data_path, unknown_acid):

    # data path
    # project_path = os.path.dirname(__file__)
    # data_path = project_path+r'/dataset/CG-21-204_SD1.xls'
    # print(project_path, data_path)

    df_source_train_pos_data = pd.read_excel(data_path, sheet_name='Train_pos_data')
    df_all_source_train_data = pd.read_excel(data_path, sheet_name='Training data')
    df_source_test_pos_data = pd.read_excel(data_path, sheet_name='Test pos_data')
    df_source_test_neg_data = pd.read_excel(data_path, sheet_name="Test_neg_data")

    """
    # move invalid data 
    """
    df_proper_train_pos_data = delete_wrong_data(df_source_train_pos_data, "train_pos_data")

    df_all_proper_train_data = delete_wrong_data(df_all_source_train_data, "all_train_data")

    df_proper_test_pos_data = delete_wrong_data(df_source_test_pos_data, "test_pos_data")

    df_proper_test_neg_data = delete_wrong_data(df_source_test_neg_data, "test_neg_data")
    """
    # generate train data
    """
    df_train_pos_data = format_data(df_proper_train_pos_data, 1)

    df_source_train_neg_data = df_all_proper_train_data[df_all_proper_train_data['Glutarylated?'] == 'No']
    df_train_neg_data = format_data(df_source_train_neg_data, 0)

    df_train_data = pd.concat([df_train_pos_data, df_train_neg_data], axis=0, ignore_index=True)

    """
    # generate independent test data
    """
    df_test_pos_data = format_data(df_proper_test_pos_data, 1)
    df_test_neg_data = format_data(df_proper_test_neg_data, 0)
    df_test_data = pd.concat([df_test_pos_data, df_test_neg_data], axis=0, ignore_index=True)
    #     display(df_train_data, df_test_data)
    return df_train_data, df_test_data

