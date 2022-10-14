import pandas as pd
import os
from IPython.core.display_functions import display
import data_processing as dp
import numpy as np
from sklearn import preprocessing
from keras.models import Model, load_model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
import tensorflow as tf
import keras.backend as K
import sklearn.metrics as metrics

# focal loss function
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.2, name="FocalLoss", **kwargs):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = float(y_true)
        bce = tf.losses.binary_crossentropy(y_true, y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        loss = tf.reduce_sum(alpha_factor * modulating_factor * bce, axis=-1)
        return loss

def MCC(y_true, y_pred):
    y_true = float(y_true)
    TP, FP, FN, TN = 0.0, 0.0, 0.0, 0.0

    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] > 0.5:
                TP += 1
            else:
                FN += 1
        if y_true[i] == 0:
            if y_pred[i] > 0.5:
                FP += 1
            else:
                TN += 1

    mcc = ((TP * TN) - (FP * FN)) / K.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-06)

    return mcc

# evaluation param
def show_performance(y_true, y_pred):
    TP, FP, FN, TN = 0.0, 0.0, 0.0, 0.0

    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] > 0.5:
                TP += 1
            else:
                FN += 1
        if y_true[i] == 0:
            if y_pred[i] > 0.5:
                FP += 1
            else:
                TN += 1


    Sn = TP / (TP + FN + K.epsilon())
    Sp = TN / (FP + TN + K.epsilon())
    Acc = (TP + TN) / len(y_true)
    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-06)

    return Sn, Sp, Acc, MCC


def get_feas_from_file(data, fea_file_path):
    df_all_fea = pd.DataFrame()
    try:
        df_all_fea = pd.read_csv(fea_file_path).iloc[:, 1:]
        display(df_all_fea)
        b_got_fea = True
    except FileNotFoundError:
        print('file{} not found!'.format(fea_file_path))
        b_got_fea = False

    if b_got_fea:
        if df_all_fea.shape[0] != data.shape[0]:
            print('error! data not matched! fea:{}, data{}'.format(df_all_fea.shape, data.shape))
            print('file path:{}'.format(fea_file_path))
            return None
        else:
            return df_all_fea
    else:
        return None


def get_data(data_path):
    return dp.get_processed_data(data_path, unknown_acid='X')


def model_pred(model_path, test, test_label, result_info_path):
    # load model
    model = load_model(model_path + os.sep
                       + 'dense_block_model.h5',
                       custom_objects={'FocalLoss': FocalLoss, 'MCC': MCC})

    # pred value
    test_pred = model.predict(test, verbose=1)
    # save pred value
    pd.DataFrame(test_pred).to_csv(result_info_path + os.sep + 'test_label_pred.csv', index=False)

    # pred result
    # get Sn, Sp, Acc, MCC, AUC
    Sn, Sp, Acc, mcc = show_performance(test_label, test_pred)
    AUC = roc_auc_score(test_label, test_pred)
    fpr1, tpr1, threshold1 = metrics.roc_curve(test_label, test_pred)

    roc_auc1 = metrics.auc(fpr1, tpr1)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr1, tpr1, color='darkorange',
             lw=lw, label='AUC = %0.2f' % roc_auc1)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    # plt.title('ROCs for Densenet')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(result_info_path + os.sep + 'test_roc.jpg')

    print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f' % (Sn, Sp, Acc, mcc, AUC))


project_path = os.path.dirname(__file__)
data_path = project_path+r'/dataset/CG-21-204_SD1.xls'

train_data, indtest_data = get_data(data_path)
train_label = train_data.label
indtest_label = indtest_data.label
all_data = pd.concat([train_data, indtest_data], axis=0, ignore_index=True)
print(train_data, indtest_data, all_data)


feature_file_path = project_path + os.sep + 'feature_method'

# features file path
one_hot_file_path = feature_file_path + os.sep + 'all_one_hot_fea.csv'
EAAC_file_path = feature_file_path+os.sep+'all_EAAC_fea(5).csv'
K_spaced_file_path = feature_file_path+os.sep+'all_K_spaced_fea(9).csv'


# get features from file
df_all_fea_one_hot = get_feas_from_file(all_data, one_hot_file_path)
df_all_fea_EAAC = get_feas_from_file(all_data, EAAC_file_path)
df_all_fea_K_spaced = get_feas_from_file(all_data, K_spaced_file_path)
print(df_all_fea_one_hot, df_all_fea_EAAC, df_all_fea_K_spaced)

# convert features to 3D
len_one_hot = df_all_fea_one_hot.shape[1]
narr_one_hot_3D_fea = np.reshape(np.array(df_all_fea_one_hot), (len(all_data), -1, 21))

len_EAAC = df_all_fea_EAAC.shape[1]
narr_EAAC_3D_fea = np.reshape(np.array(df_all_fea_EAAC), (len(all_data), -1, 21))

len_ks = df_all_fea_K_spaced.shape[1]
narr_kspaced_3D_fea = np.reshape(np.array(df_all_fea_K_spaced), (len(all_data), -1, 21))

# concat and normalization
# df_fea = pd.concat([df_all_fea_one_hot, df_all_fea_EAAC], axis=1)
# df_fea = pd.concat([df_all_fea_EAAC, df_all_fea_K_spaced], axis=1)
# df_fea = pd.concat([df_all_fea_one_hot, df_all_fea_EAAC, df_all_fea_K_spaced], axis=1)
# narr_fea = preprocessing.scale(df_fea, axis=1, with_mean=True, with_std=True)
# narr_one_hot_3D_fea_nor = np.reshape(narr_fea[:, :len_one_hot], (len(all_data), -1, 21))
# narr_EAAC_3D_fea_nor = np.reshape(narr_fea[:, len_one_hot:len_one_hot+len_EAAC], (len(all_data), -1, 21))
# narr_kspaced_3D_fea_nor = np.reshape(narr_fea[:, len_one_hot+len_EAAC:], (len(all_data), -1, 21))


# narr_all_features = np.concatenate([narr_one_hot_3D_fea_nor,
#                                     narr_EAAC_3D_fea_nor,
#                                     narr_kspaced_3D_fea_nor], axis=1)

narr_all_features = narr_one_hot_3D_fea

narr_train_features = narr_all_features[0:train_data.shape[0]]
narr_indep_test_features = narr_all_features[train_data.shape[0]:]


model_path = project_path + os.sep + 'model'
result_path = project_path + os.sep + 'result'
model_pred(model_path, narr_indep_test_features, indtest_label, result_path)





