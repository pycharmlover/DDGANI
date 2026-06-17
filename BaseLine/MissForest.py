import numpy as np
import pandas as pd
import sklearn.neighbors._base
import sys

from util import categorical_to_code

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
import Mean
import logging
logging.basicConfig(level=logging.WARNING)

from sklearn.preprocessing import LabelEncoder


def fill_data_missForest(data_m, nan_data, con_cols, cat_cols, tree_num):
    fill_mean_data = Mean.fill_data_mean(nan_data, con_cols)
    copy_ori_data = fill_mean_data.copy()
    copy_ori_data = pd.DataFrame(copy_ori_data)
    miss_code, enc = categorical_to_code(copy_ori_data, cat_cols, enc=None)
    miss_code[data_m == 0] = np.nan
    imputer = MissForest(verbose=0,criterion='squared_error', max_features=1.0, n_estimators=tree_num)
    data_imputed = imputer.fit_transform(miss_code)
    data_imputed[cat_cols] = data_imputed[cat_cols].round().astype(int)
    data_imputed = pd.DataFrame(data_imputed)
    if len(cat_cols) != 0:
        data_imputed[cat_cols] = enc.inverse_transform(data_imputed[cat_cols])
    miss_data = data_imputed.values
    return miss_data

# def fill_data_missForest(data_m, nan_data, con_cols, cat_cols):
#     # 用均值填补数值型数据，作为初始估计
#     fill_mean_data = Mean.fill_data_mean(nan_data, con_cols)
#     # 创建数据的副本
#     copy_ori_data = fill_mean_data.copy()
#     copy_ori_data = pd.DataFrame(copy_ori_data)

#     # 对类别数据进行编码
#     miss_code, enc = categorical_to_code(copy_ori_data, cat_cols, enc=None)
#     miss_code[data_m == 0] = np.nan

#     # 使用 MissForest 进行缺失值填补
#     imputer = MissForest(verbose=0, criterion='squared_error', max_features=1.0, n_estimators=100)
#     data_imputed = imputer.fit_transform(miss_code)

#     # 恢复类别数据为原始类别值
#     if len(cat_cols) != 0:
#         for col in cat_cols:
#             # 先将插补后的数据转换为整数
#             data_imputed[:, col] = np.round(data_imputed[:, col]).astype(int)
#             data_imputed[:, col] = data_imputed[:, col].astype(int)
#             # 再恢复原始类别
#             data_imputed[:, col] = enc[col].inverse_transform(data_imputed[:, col])
#     return data_imputed