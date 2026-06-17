import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
sys.path.extend(['/home/Xianger123/DDGANI/BaseLine/holoclean'])
from BaseLine.holoclean.detect.nulldetector import NullDetector
from BaseLine.holoclean.repair.featurize.freqfeat import FreqFeaturizer
from BaseLine.holoclean.repair.featurize.occurattrfeat import OccurAttrFeaturizer
from BaseLine.holoclean import holoclean
from BaseLine.holoclean.detect import *
from BaseLine.holoclean.repair.featurize import *
import time

def fill_data_holo(data_m, nan_data, ori_data, continuous_cols, categorical_cols):
    print("开始时间：", time.time())
    dirty_data = pd.DataFrame(nan_data.copy())
    # clean_data = pd.DataFrame(ori_data.copy())
    dirty_path = '/home/Xianger123/DDGANI/BaseLine/holoclean/holoclean_dataset/dirty.csv'
    # clean_path = '/home/Xianger123/DDGANI/BaseLine/holoclean/holoclean_dataset/clean.csv'
    dirty_data.to_csv(dirty_path, index=False)
    # clean_data.to_csv(clean_path, index=False)
    hc = holoclean.HoloClean(
    db_name='holo',
    domain_thresh_1=0.0,
    domain_thresh_2=0.0,
    weak_label_thresh=0.99,
    max_domain=10000,
    cor_strength=0.6,
    nb_cor_strength=0.8,
    weight_decay=0.01,
    learning_rate=0.001,
    threads=1,
    batch_size=1,
    verbose=True,
    timeout=3 * 60000,
    print_fw=True,
    ).session
    
    if continuous_cols is not None:
        continuous_cols_str = [str(col) for col in continuous_cols]
    else:
        continuous_cols_str = None
    # 2. Load training data and denial constraints.
    hc.load_data('null_data', dirty_path, continuous_cols_str)
    # hc.load_dcs('../testdata/hospital/hospital_constraints.txt')
    # hc.ds.set_constraints(hc.get_dcs())

    # 3. Detect erroneous cells using these two detectors.
    detectors = [NullDetector()]
    hc.detect_errors(detectors)

    # 4. Repair errors utilizing the defined features.
    hc.generate_domain()
    hc.run_estimator()
    featurizers = [
        OccurAttrFeaturizer(),
        FreqFeaturizer(),
        # ConstraintFeaturizer(),
    ]
    try:
        repaired_data = hc.repair_errors(featurizers)
        fill_data_np = repaired_data.values[:, 1:]
    except:
        print('question')
        fill_data_np = nan_data
    miss_data = pd.DataFrame(nan_data)
    for i in range(miss_data.shape[1]):
        if i in categorical_cols:
            attr_list_map = miss_data[miss_data.columns[i]].value_counts()
            for index, tuple in enumerate(fill_data_np):
                if tuple[i] not in attr_list_map.index.tolist() or str(tuple[i]) == '_nan_':
                    tuple[i] = attr_list_map.index.tolist()[0]
                fill_data_np[index, i] = tuple[i]
        else:
            all = 0
            for j in nan_data[:, i]:
                if not np.isnan(j):
                    all = all + j
            mean = all / nan_data.shape[0]
            for index, o in enumerate(fill_data_np[:, i]):
                if str(o) == '_nan_':
                    fill_data_np[index, i] = 0
                else:
                    fill_data_np[index, i] = float(o)
    return fill_data_np

    # 5. Evaluate the correctness of the results.
    # report = hc.evaluate(fpath='../testdata/hospital/hospital_clean.csv',
    #             tid_col='tid',
    #             attr_col='attribute',
    #             val_col='correct_val')
def fill_data_holo_llm(nan_data, ori_data, data_m, missing_cols):
    dirty_data = pd.DataFrame(nan_data.copy())
    # clean_data = pd.DataFrame(ori_data.copy())
    dirty_path = '/home/Xianger123/DDGANI/BaseLine/holoclean/holoclean_dataset/dirty.csv'
    # clean_path = '/home/Xianger123/DDGANI/BaseLine/holoclean/holoclean_dataset/clean.csv'
    dirty_data.to_csv(dirty_path, index=False)
    # clean_data.to_csv(clean_path, index=False)
    hc = holoclean.HoloClean(
    db_name='holo',
    domain_thresh_1=0.0,
    domain_thresh_2=0.0,
    weak_label_thresh=0.99,
    max_domain=10000,
    cor_strength=0.6,
    nb_cor_strength=0.8,
    weight_decay=0.01,
    learning_rate=0.001,
    threads=1,
    batch_size=1,
    verbose=True,
    timeout=3 * 60000,
    print_fw=True,
    ).session

    # 2. Load training data and denial constraints.
    hc.load_data('null_data', dirty_path)
    # hc.load_dcs('../testdata/hospital/hospital_constraints.txt')
    # hc.ds.set_constraints(hc.get_dcs())

    # 3. Detect erroneous cells using these two detectors.
    detectors = [NullDetector()]
    hc.detect_errors(detectors)

    # 4. Repair errors utilizing the defined features.
    hc.generate_domain()
    hc.run_estimator()
    featurizers = [
        OccurAttrFeaturizer(),
        FreqFeaturizer(),
        # ConstraintFeaturizer(),
    ]

    repaired_data = hc.repair_errors(featurizers)
    fill_data_np = repaired_data.values[:, 1:]
    miss_data = pd.DataFrame(nan_data)
    for i in range(miss_data.shape[1]):
        if i in missing_cols:
            attr_list_map = miss_data[miss_data.columns[i]].value_counts()
            for index, tuple in enumerate(fill_data_np):
                if tuple[i] not in attr_list_map.index.tolist() or str(tuple[i]) == '_nan_':
                    tuple[i] = attr_list_map.index.tolist()[-1]
                fill_data_np[index, i] = tuple[i]
    fill_data_np = pd.DataFrame(fill_data_np)
    acc = impute_acc(fill_data_np, ori_data, data_m, missing_cols)
    return acc

    # 5. Evaluate the correctness of the results.
    # report = hc.evaluate(fpath='../testdata/hospital/hospital_clean.csv',
    #             tid_col='tid',
    #             attr_col='attribute',
    #             val_col='correct_val')

def impute_acc(imputed_data, ori_data, M, missing_cols):
    imputed_data = imputed_data.values
    ori_data = ori_data.values
    accuracies = []
    for col in missing_cols:
        # 获取这一列的缺失标记，即M中为0的位置
        missing_mask = (M[:, col] == 0)
        # 计算填充的准确率
        # 这里假设ori_data和imputed_data都已经正确处理了相应的数据类型（如分类数据已经编码等）
        if np.sum(missing_mask) > 0:  # 避免除以0
            accuracy = np.mean(imputed_data[missing_mask, col] == ori_data[missing_mask, col])
            accuracies.append(accuracy)
    # 计算所有缺失列的平均准确率
    if accuracies:  # 避免空列表的情况
        average_accuracy = np.mean(accuracies)
    else:
        average_accuracy = np.nan  # 如果没有缺失列或无有效缺失，返回NaN
    return average_accuracy
