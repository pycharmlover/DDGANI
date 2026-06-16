# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils.util import reconvert_data
from collections import Counter, defaultdict
import itertools

# Global CFD candidate pool: generated during CFD mining; the refine stage only queries this pool and does not build candidates on the fly.
GLOBAL_CFD_CANDIDATE_POOL = []
GLOBAL_CFD_MIN_SUPPORT = 100
GLOBAL_CFD_MIN_OBSERVED_SUPPORT = 100
GLOBAL_CFD_MIN_X_MIS_COUNT = 0


class CFDModel(nn.Module):
    def __init__(self, input_size, output_size, x_index_list, y_index):
        super(CFDModel, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, output_size)
        self.x_index_list = x_index_list
        self.y_index = y_index

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def set(self, new_x_index_list):
        self.x_index_list = new_x_index_list


def freeze_rule_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_rule_model(model):
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    return model


def build_begin_list(fields):
    begin_list = [0]
    begin = 0
    for field in fields:
        if field.data_type == "Categorical Data":
            begin += len(field.dict)
        else:
            begin += 1
        begin_list.append(begin)
    return begin_list


def build_rule_input(code, x_index_list, y_index, fields):
    begin_list = build_begin_list(fields)
    x_parts = []
    for x_idx in range(len(fields)):
        if x_idx in x_index_list:
            x_parts.append(code[:, begin_list[x_idx]:begin_list[x_idx + 1]])
        elif x_idx != y_index:
            width = begin_list[x_idx + 1] - begin_list[x_idx]
            x_parts.append(torch.zeros((code.shape[0], width), dtype=code.dtype, device=code.device))
    if len(x_parts) == 0:
        return torch.zeros((code.shape[0], 0), dtype=code.dtype, device=code.device), begin_list
    return torch.cat(x_parts, dim=1), begin_list


def get_mode_label_index(code, begin_list, attr_index, rows):
    if rows is None or len(rows) == 0:
        return None
    rows = np.array(rows, dtype=int)
    data = code[rows][:, begin_list[attr_index]:begin_list[attr_index + 1]]
    if data.shape[1] <= 1:
        return 0
    labels = torch.argmax(data, dim=1).detach().cpu().numpy()
    if len(labels) == 0:
        return None
    labels = labels.astype(int).tolist()
    return Counter(labels).most_common(1)[0][0]


def split_rule_models(model_list):
    CFD_models = []
    CFD_models = []
    for model in model_list:
        if getattr(model, "rule_kind", "FD") == "CFD":
            CFD_models.append(model)
        else:
            CFD_models.append(model)
    return CFD_models, CFD_models


def train_Model(x, y, model):
    if x is None or y is None or len(x) == 0:
        return model

    unfreeze_rule_model(model)
    train_dataset = TensorDataset(x, y)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02)
    for epoch in range(300):
        for batch_features, batch_labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward(retain_graph=True)
            optimizer.step()
        if epoch % 100 == 0:
            acc = test_model(train_dataloader, model)
            if acc == 1:
                break
    return model


def test_model(data, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_features, batch_labels in data:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    accuracy = correct / total
    return accuracy

def entropy(labels):
    if len(labels) == 0:
        return 0
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def get_my_CFD_loss_trainable(CFD_model_list, decode_code, fields):
    if len(CFD_model_list) == 0:
        return 0
    criterion = nn.CrossEntropyLoss()
    all_loss = 0
    valid_num = 0
    for CFD_model in CFD_model_list:
        x, begin_list = build_rule_input(decode_code, CFD_model.x_index_list, CFD_model.y_index, fields)
        y_code = decode_code[:, begin_list[CFD_model.y_index]:begin_list[CFD_model.y_index + 1]]
        if y_code.shape[1] <= 1:
            continue
        y = torch.argmax(y_code, dim=1).long()
        outputs = CFD_model(x)
        all_loss = all_loss + criterion(outputs, y)
        valid_num += 1
    if valid_num == 0:
        return 0
    return all_loss / valid_num

def _get_cfd_soft_pattern_weight(model, decode_code, fields, begin_list):
    weight = torch.ones(decode_code.shape[0], dtype=decode_code.dtype, device=decode_code.device)
    x_pattern = getattr(model, 'x_pattern', ['_'] * len(model.x_index_list))
    x_pattern_label_indices = getattr(model, 'x_pattern_label_indices', [None] * len(model.x_index_list))
    for pos, x_idx in enumerate(model.x_index_list):
        if pos >= len(x_pattern):
            continue
        if x_pattern[pos] == '_':
            continue
        label_idx = None
        if pos < len(x_pattern_label_indices):
            label_idx = x_pattern_label_indices[pos]
        if label_idx is None:
            continue
        x_code = decode_code[:, begin_list[x_idx]:begin_list[x_idx + 1]]
        if x_code.shape[1] <= 1:
            continue
        if label_idx >= x_code.shape[1]:
            continue
        weight = weight * x_code[:, int(label_idx)]
    return weight

def get_my_CFD_loss(CFD_model_list, decode_code, fields):
    if len(CFD_model_list) == 0:
        return 0
    criterion = nn.CrossEntropyLoss(reduction='none')
    all_loss = 0
    valid_num = 0
    begin_list = build_begin_list(fields)
    for CFD_model in CFD_model_list:
        x, _ = build_rule_input(decode_code, CFD_model.x_index_list, CFD_model.y_index, fields)
        y_code = decode_code[:, begin_list[CFD_model.y_index]:begin_list[CFD_model.y_index + 1]]
        if y_code.shape[1] <= 1:
            continue
        outputs = CFD_model(x)
        row_weight = _get_cfd_soft_pattern_weight(CFD_model, decode_code, fields, begin_list)
        if getattr(CFD_model, 'y_pattern', '_') == '_':
            targets = torch.argmax(y_code, dim=1).long()
        else:
            target_index = getattr(CFD_model, 'y_pattern_label_index', None)
            if target_index is None:
                continue
            targets = torch.full((decode_code.shape[0],), int(target_index), dtype=torch.long, device=decode_code.device)
        row_loss = criterion(outputs, targets)
        weight_sum = torch.sum(row_weight) + 1e-8
        if float(weight_sum.detach().cpu()) <= 1e-12:
            continue
        all_loss = all_loss + torch.sum(row_loss * row_weight) / weight_sum
        valid_num += 1
    if valid_num == 0:
        return 0
    return all_loss / valid_num

def get_CFD_model_Tree(miss_data, data_m, categorical_cols, zero_feed_data, fields, device, sort_corr_dict):
    """
    The CFD mining stage generates two types of results:
    1. Mined CFDs: observed support >= min_support, used to train CFD models directly;
    2. CFD candidate pool: observed support < min_support but support_upper_bound >= min_support,
       which the subsequent refine stage only queries by lhs -> rhs for validation.

    New support setting method:
    - min_observed_support is the manually defined minimum observed support;
    - X_mis is computed separately for each candidate CFD, namely the number of tuples that contain
      missing values on Z' ∪ {A} and may still match t'_p;
    - min_x_mis_count = min X_mis;
    - min_support = min_observed_support + min_x_mis_count.

    This no longer uses the number of rows with missing values in the whole table, avoiding an overly
    large min_support caused by too many table-level missing rows.
    """
    global GLOBAL_CFD_CANDIDATE_POOL
    global GLOBAL_CFD_MIN_SUPPORT
    global GLOBAL_CFD_MIN_OBSERVED_SUPPORT
    global GLOBAL_CFD_MIN_X_MIS_COUNT

    value_cat = []
    col_dict = {}
    data = miss_data.values
    values = miss_data.columns
    has_data_index = []
    observer_data = []
    for col_index, col_val in enumerate(data_m.T):
        cur_has_data_index = []
        cur_observer_data = []
        for row_index, val in enumerate(col_val):
            if val == 1:
                cur_has_data_index.append(row_index)
                cur_observer_data.append(data[row_index][col_index])
        has_data_index.append(cur_has_data_index)
        observer_data.append(cur_observer_data)

    encoder_data = []
    new_sort_corr_dict = {}
    for col_index, col_val in enumerate(data.T):
        arr = np.array(col_val)
        cur_corr_sort = []
        if col_index in categorical_cols:
            arr = arr.astype(np.str_)
            unique_values, encoded_arr = np.unique(arr, return_inverse=True)
            encoded_arr += 1
            value_cat.append(miss_data.columns[col_index])
        else:
            bins = np.array_split(np.sort(arr), 10)
            encoded_arr = np.zeros_like(arr)
            for i, bin_ in enumerate(bins):
                encoded_arr[np.isin(arr, bin_)] = i
        col_dict[miss_data.columns[col_index]] = col_index
        col_index_corr_sort = sort_corr_dict[col_index]
        for i in col_index_corr_sort:
            if i in categorical_cols and i != col_index:
                cur_corr_sort.append(i)
        new_sort_corr_dict[col_index] = cur_corr_sort
        encoder_data.append(encoded_arr)

    models = []
    cfd_rule_list = []
    cfd_candidate_pool = []
    seen_rule = set()
    seen_pool_rule = set()

    # ----------------------------------------------------------
    # Defined minimum observed support.
    # The final min_support is automatically set to: minimum observed support + min X_mis.
    # X_mis is computed separately for each candidate CFD.
    # ----------------------------------------------------------
    min_observed_support = 225
    max_lhs_size = 3

    raw_candidates_by_observed_support = defaultdict(list)
    raw_seen_rules = set()
    min_x_mis_count = None

    for y_index in categorical_cols:
        lhs_expand_pool = [idx for idx in new_sort_corr_dict.get(y_index, []) if idx != y_index]
        candidate_x_index_list = generate_candidate_x_sets(lhs_expand_pool, max_size=max_lhs_size)
        for x_index in candidate_x_index_list:
            cfd_candidates = get_cfd_candidates_by_tree(
                x_index, y_index, miss_data, has_data_index, min_support=min_observed_support
            )
            for candidate in cfd_candidates:
                rule = candidate['cfd_rule']
                if rule in raw_seen_rules:
                    continue
                raw_seen_rules.add(rule)

                observed_support = int(candidate.get('support', len(candidate.get('condition_index', []))))
                x_mis_count = get_cfd_x_mis_count(
                    miss_data=miss_data,
                    data_m=data_m,
                    candidate=candidate
                )
                support_upper_bound = observed_support + int(x_mis_count)

                candidate['observed_support'] = int(observed_support)
                candidate['x_mis_count'] = int(x_mis_count)
                candidate['support_upper_bound'] = int(support_upper_bound)
                candidate['min_observed_support'] = int(min_observed_support)
                candidate['lhs_expand_pool'] = list(lhs_expand_pool)
                candidate['max_lhs_size'] = int(max_lhs_size)

                min_x_mis_count = int(x_mis_count) if min_x_mis_count is None else min(min_x_mis_count, int(x_mis_count))
                raw_candidates_by_observed_support[int(observed_support)].append(candidate)

    # ----------------------------------------------------------
    # 2. Use the minimum X_mis among all candidate CFDs to automatically determine the actual min_support.
    # ----------------------------------------------------------
    print(f"最小miss数量为：{0 if min_x_mis_count is None else min_x_mis_count}")
    min_support = int(min_observed_support + (0 if min_x_mis_count is None else min_x_mis_count))
    min_x_mis_value = int(0 if min_x_mis_count is None else min_x_mis_count)

    GLOBAL_CFD_MIN_OBSERVED_SUPPORT = int(min_observed_support)
    GLOBAL_CFD_MIN_X_MIS_COUNT = int(min_x_mis_value)
    GLOBAL_CFD_MIN_SUPPORT = int(min_support)

    # ----------------------------------------------------------
    # 3. Sort by observed_support keys and filter by segments:
    #    - key >= min_support: all candidates in this segment are official CFD candidates;
    #    - min_observed_support <= key < min_support: only support_upper_bound needs to be checked.
    # ----------------------------------------------------------
    sorted_observed_support_keys = sorted(raw_candidates_by_observed_support.keys(), reverse=True)
    official_support_keys = [key for key in sorted_observed_support_keys if key >= min_support]
    upper_bound_pool_support_keys = [
        key for key in sorted_observed_support_keys
        if min_observed_support <= key < min_support
    ]

    for support_key in official_support_keys:
        for candidate in raw_candidates_by_observed_support[support_key]:
            rule = candidate['cfd_rule']
            if rule in seen_rule:
                continue

            x_index = list(candidate['x_index_list'])
            y_index = int(candidate['y_index'])
            lhs_expand_pool = list(candidate.get('lhs_expand_pool', []))
            x_mis_count = int(candidate.get('x_mis_count', 0))
            support_upper_bound = int(candidate.get('support_upper_bound', support_key + x_mis_count))

            candidate['min_support'] = int(min_support)
            candidate['min_x_mis_count'] = int(min_x_mis_value)

            model = get_model_by_tree_condition(
                x_index, y_index, zero_feed_data, has_data_index, fields, device, candidate['condition_index']
            )
            if model is not None:
                attach_cfd_metadata(
                    model, candidate, zero_feed_data, fields,
                    lhs_expand_pool=lhs_expand_pool,
                    max_lhs_size=max_lhs_size,
                    min_support=min_support
                )
                model.min_observed_support = int(min_observed_support)
                model.x_mis_count = int(x_mis_count)
                model.min_x_mis_count = int(min_x_mis_value)
                model.support_upper_bound = int(support_upper_bound)
                freeze_rule_model(model)
                models.append(model)
                cfd_rule_list.append(rule)
                seen_rule.add(rule)

    for support_key in upper_bound_pool_support_keys:
        for candidate in raw_candidates_by_observed_support[support_key]:
            rule = candidate['cfd_rule']
            if rule in seen_pool_rule:
                continue

            x_mis_count = int(candidate.get('x_mis_count', 0))
            support_upper_bound = int(candidate.get('support_upper_bound', support_key + x_mis_count))
            candidate['min_support'] = int(min_support)
            candidate['min_x_mis_count'] = int(min_x_mis_value)

            if support_upper_bound >= min_support:
                seen_pool_rule.add(rule)
                cfd_candidate_pool.append(candidate)

    cfd_candidate_pool = deduplicate_candidate_dicts(cfd_candidate_pool)
    GLOBAL_CFD_CANDIDATE_POOL = list(cfd_candidate_pool)

    for model in models:
        model.cfd_candidate_pool = list(cfd_candidate_pool)
        model.min_observed_support = int(min_observed_support)
        model.min_x_mis_count = int(0 if min_x_mis_count is None else min_x_mis_count)
        model.min_support = int(min_support)

    save_cfd_rules(cfd_rule_list)
    save_cfd_rules(
        [cand['cfd_rule'] for cand in cfd_candidate_pool],
        save_dir='out/CFD_list/',
        file_name='support_upper_bound_cfd_rules_new.txt'
    )
    return models

def get_model_by_tree_condition(x_index, y_index, zero_feed_data, has_data_index, fields, device, condition_index):
    set_index = set(has_data_index[y_index])
    for x in x_index:
        set_index = set_index.intersection(set(has_data_index[x]))
    set_index = set_index.intersection(set(condition_index))
    set_index = np.array(list(set_index))

    if len(set_index) == 0:
        return None

    X = []
    begin_list = [0]
    begin = 0
    for index, field in enumerate(fields):
        if field.data_type == "Categorical Data":
            begin += len(field.dict)
        else:
            begin += 1
        begin_list.append(begin)

    for x in range(len(fields)):
        if x in x_index:
            cur_data = zero_feed_data[set_index]
            cur_data = cur_data[:, begin_list[x]:begin_list[x+1]]
            X.append(cur_data)
        elif x != y_index:
            cur_data = torch.zeros((len(set_index), begin_list[x+1] - begin_list[x]), dtype=zero_feed_data.dtype).to(device)
            X.append(cur_data)
    X = torch.cat(X, dim=1).to(device)
    Y = zero_feed_data[set_index]
    Y = Y[:, begin_list[y_index]:begin_list[y_index+1]]
    Y = torch.argmax(Y, dim=1).long().to(device)

    if len(set_index) == 0:
        return None

    input_dim = X.shape[1]
    output_dim = begin_list[y_index+1] - begin_list[y_index]
    model = CFDModel(input_dim, output_dim, x_index, y_index).to(device)
    model = train_Model(X, Y, model)
    return model

def generate_candidate_x_sets(candidate_cols, max_size=2):
    result = []
    if candidate_cols is None:
        return result
    unique_cols = []
    for col in candidate_cols:
        if col not in unique_cols:
            unique_cols.append(col)
    upper = min(max_size, len(unique_cols))
    for size in range(1, upper + 1):
        for combo in itertools.combinations(unique_cols, size):
            result.append(list(combo))
    return result


def get_cfd_candidates_by_tree(x_index, y_index, miss_data, has_data_index, min_support):
    set_index = set(has_data_index[y_index])
    for x in x_index:
        set_index = set_index.intersection(set(has_data_index[x]))
    set_index = np.array(sorted(list(set_index)), dtype=int)

    if len(set_index) == 0:
        return []

    x_names = [miss_data.columns[x] for x in x_index]
    y_name = miss_data.columns[y_index]
    results = []

    if is_variable_cfd(
        x_index, y_index, miss_data, set_index,
        min_group_support=min_support
    ):
        results.append(build_cfd_candidate(
            x_index=x_index,
            y_index=y_index,
            x_names=x_names,
            y_name=y_name,
            x_pattern=['_'] * len(x_index),
            y_pattern='_',
            domain_rows=set_index,
            condition_index=set_index,
            cfd_type='variable',
            support=len(set_index),
            conf=1.0,
        ))
        return results

    pattern_dict = collect_x_patterns(x_index, miss_data, set_index)
    for x_pattern, row_ids in pattern_dict.items():
        row_ids = np.array(sorted(set(row_ids)), dtype=int)
        if len(row_ids) < min_support:
            continue

        if is_variable_cfd(x_index, y_index, miss_data, row_ids, min_group_support=min_support):
            results.append(build_cfd_candidate(
                x_index=x_index,
                y_index=y_index,
                x_names=x_names,
                y_name=y_name,
                x_pattern=list(x_pattern),
                y_pattern='_',
                domain_rows=row_ids,
                condition_index=row_ids,
                cfd_type='pattern_variable_rhs',
                support=len(row_ids),
                conf=1.0,
            ))
            continue

        y_values = miss_data.iloc[row_ids, y_index].astype(str).values
        counter = Counter(y_values)
        for y_val, cnt in counter.most_common():
            conf = cnt / len(row_ids)
            if cnt < min_support:
                continue
            condition_index = row_ids[y_values == y_val]
            results.append(build_cfd_candidate(
                x_index=x_index,
                y_index=y_index,
                x_names=x_names,
                y_name=y_name,
                x_pattern=list(x_pattern),
                y_pattern=str(y_val),
                domain_rows=row_ids,
                condition_index=condition_index,
                cfd_type='pattern_constant_rhs',
                support=len(condition_index),
                conf=conf,
            ))

    return deduplicate_cfd_candidates(results)


def build_cfd_candidate(x_index, y_index, x_names, y_name, x_pattern, y_pattern,
                        domain_rows, condition_index, cfd_type, support, conf,
                        lhs_expand_pool=None, refine_depth=0):
    return {
        'x_index_list': list(x_index),
        'y_index': y_index,
        'x_pattern': list(x_pattern),
        'y_pattern': str(y_pattern),
        'domain_rows': np.array(sorted(set(domain_rows)), dtype=int),
        'condition_index': np.array(sorted(set(condition_index)), dtype=int),
        'cfd_type': cfd_type,
        'support': int(support),
        'conf': float(conf),
        'lhs_expand_pool': [] if lhs_expand_pool is None else list(lhs_expand_pool),
        'refine_depth': refine_depth,
        'cfd_rule': format_cfd_rule(x_names, y_name, x_pattern, y_pattern),
    }


def attach_cfd_metadata(model, candidate, zero_feed_data, fields,
                        lhs_expand_pool=None, max_lhs_size=3,
                        min_support=20):
    begin_list = build_begin_list(fields)
    model.rule_kind = 'CFD'
    model.cfd_type = candidate['cfd_type']
    model.cfd_rule = candidate['cfd_rule']
    model.x_pattern = list(candidate['x_pattern'])
    model.y_pattern = str(candidate['y_pattern'])
    model.min_support = int(min_support)
    model.max_lhs_size = int(max_lhs_size)
    model.lhs_expand_pool = list(lhs_expand_pool if lhs_expand_pool is not None else candidate.get('lhs_expand_pool', []))
    model.refine_depth = int(candidate.get('refine_depth', 0))
    model.domain_rows = np.array(candidate.get('domain_rows', []), dtype=int)
    model.condition_index = np.array(candidate.get('condition_index', []), dtype=int)
    model.support = int(candidate.get('support', len(model.condition_index)))
    model.conf = float(candidate.get('conf', 0.0))
    model.x_pattern_label_indices = []
    for pos, x_idx in enumerate(model.x_index_list):
        if pos >= len(model.x_pattern) or model.x_pattern[pos] == '_':
            model.x_pattern_label_indices.append(None)
        else:
            rows = model.domain_rows if len(model.domain_rows) > 0 else model.condition_index
            model.x_pattern_label_indices.append(get_mode_label_index(zero_feed_data, begin_list, x_idx, rows))
    if model.y_pattern == '_':
        model.y_pattern_label_index = None
    else:
        model.y_pattern_label_index = get_mode_label_index(
            zero_feed_data, begin_list, model.y_index,
            model.condition_index if len(model.condition_index) > 0 else model.domain_rows
        )


def collect_x_patterns(x_index, miss_data, set_index):
    pattern_dict = defaultdict(list)
    for row_idx in set_index:
        x_values = [str(miss_data.iloc[row_idx, x]) for x in x_index]
        pattern_list = generate_patterns_from_values(x_values)
        for pattern in pattern_list:
            if all(v == '_' for v in pattern):
                continue
            pattern_dict[tuple(pattern)].append(int(row_idx))
    return pattern_dict


def generate_patterns_from_values(values):
    result = []
    k = len(values)
    for mask in itertools.product([0, 1], repeat=k):
        pattern = []
        for i in range(k):
            if mask[i] == 1:
                pattern.append(str(values[i]))
            else:
                pattern.append('_')
        result.append(pattern)
    return result


def format_cfd_rule(x_names, y_name, x_pattern, y_pattern):
    lhs = str(list(x_names))
    pattern_part = []
    for name, val in zip(x_names, x_pattern):
        pattern_part.append(f"{name}={val}")
    pattern_part.append(f"{y_name}={y_pattern}")
    return f"{lhs} ----> {y_name} | " + ",".join(pattern_part)


def is_variable_cfd(x_index, y_index, miss_data, set_index, min_group_support):
    group_dict = defaultdict(list)
    for row_idx in set_index:
        x_val = tuple(miss_data.iloc[row_idx, x_index].astype(str).values)
        y_val = str(miss_data.iloc[row_idx, y_index])
        group_dict[x_val].append(y_val)

    repeated_group_num = 0
    repeated_sample_num = 0

    for _, y_list in group_dict.items():
        # A group participates in validation only if it reaches at least min_group_support.
        if len(y_list) >= min_group_support:
            repeated_group_num += 1
            repeated_sample_num += len(y_list)

            # Multiple Y values in the same X group indicate a violation of the variable CFD / FD.
            if len(set(y_list)) > 1:
                return False

    # At least one support-satisfying group must actually participate in validation before the rule can be accepted.
    return repeated_group_num > 0


def deduplicate_cfd_candidates(results):
    new_results = []
    seen = set()
    for candidate in results:
        if candidate['cfd_rule'] not in seen:
            seen.add(candidate['cfd_rule'])
            new_results.append(candidate)
    return new_results


def append_support_ub_cfd_rule(rule,
                               save_dir='out/CFD_list/',
                               file_name='support_upper_bound_cfd_rules.txt'):
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, file_name)
    rule = str(rule)

    old_rules = set()
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            old_rules = set(line.strip() for line in f if line.strip())

    if rule in old_rules:
        return

    with open(save_path, 'a', encoding='utf-8') as f:
        f.write(rule + '\n')

def save_cfd_rules(cfd_rule_list, save_dir='out/CFD_list/', file_name='cfd_rules_new.txt'):
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, file_name)
    new_rule_list = []
    seen = set()
    for rule in cfd_rule_list:
        if rule not in seen:
            seen.add(rule)
            new_rule_list.append(rule)

    with open(save_path, 'w', encoding='utf-8') as f:
        for rule in new_rule_list:
            f.write(rule + '\n')


def get_model_by_tree(x_index, y_index, zero_feed_data, has_data_index, fields, device):
    set_index = set(has_data_index[y_index])
    for x in x_index:
        set_index = set_index.intersection(set(has_data_index[x]))
    set_index = np.array(list(set_index))
    X = []
    begin_list = [0]
    begin = 0
    for index,field in enumerate(fields):
        if field.data_type == "Categorical Data":
            begin += len(field.dict)
        else:
            begin += 1
        begin_list.append(begin)
    for x in range(len(fields)):
        if x in x_index:
            cur_data = zero_feed_data[set_index]
            cur_data = cur_data[:, begin_list[x]:begin_list[x+1]]
            X.append(cur_data)
        elif x != y_index:
            cur_data = torch.zeros((len(set_index), begin_list[x+1] - begin_list[x]), dtype=zero_feed_data.dtype).to(device)
            X.append(cur_data)
    X = torch.cat(X, dim=1).to(device)
    Y = zero_feed_data[set_index]
    Y = Y[:, begin_list[y_index]:begin_list[y_index+1]]
    Y = torch.argmax(Y, dim=1).long().to(device)
    input_dim = X.shape[1]
    output_dim = begin_list[y_index+1] - begin_list[y_index]
    model = CFDModel(input_dim, output_dim, x_index, y_index).to(device)
    model = train_Model(X, Y, model)
    return model


def get_x_index(nested_list):
    result_list = []
    for item in nested_list:
        if isinstance(item, list):
            if len(result_list) == 0:
                for sub_item in item:
                    result_list.append([sub_item])
            else:
                sub_list = get_x_index(item)
                res_list = result_list.copy()
                cur_list = res_list.copy()
                result_list = []
                for sub_item in sub_list:
                    cur_list.append(sub_item)
                    result_list.append(cur_list)
                    cur_list = res_list.copy()
        else:
            result_list.append(item)
    return result_list

def buildTreeDFS(encoder_data, col_dict, col_name, has_data_index, featLabels, cur_choose_row_index, graph):
    for node in graph.keys():
        cur_node = []
        DFS(encoder_data, col_dict, col_name, has_data_index, featLabels, cur_choose_row_index, graph, cur_node, node)

def DFS(encoder_data, col_dict, col_name, has_data_index, featLabels, cur_choose_row_index, graph, cur_node, start_node):
    # if col_dict[col_name] == 1:
    #     print(1)
    cur_node.append(start_node)
    if isCurNodeInFDs(cur_node, featLabels):
        return
    if len(cur_choose_row_index) == 0:
        return
    if len(cur_node) == 3:
        return
    if isCurNodeFD(encoder_data, col_dict, col_name, has_data_index, cur_node):
        new_node = cur_node.copy()
        updateFeatLabels(new_node, featLabels)
        return
    for node in graph[start_node]:
        if node not in cur_node:
            start_node = node
            DFS(encoder_data, col_dict, col_name, has_data_index, featLabels, cur_choose_row_index, graph,
                         cur_node, start_node)

            cur_node.pop()

def updateFeatLabels(new_node, featLabels):
    # Check whether featLabels contains one of its subsets.
    node_list = []
    for node in featLabels:
        if set(new_node) <= set(node) and len(new_node) > 0:
            node_list.append(node)
    if len(node_list) > 0:
        for n in node_list:
            featLabels.remove(n)
    featLabels.append(new_node)

def isCurNodeInFDs(cur_node, featLabels):
    if len(featLabels) == 0 or len(cur_node) == 0:
        return False
    for node in featLabels:
        if set(node) <= set(cur_node):
            return True
    return False


def isCurNodeFD(encoder_data, col_dict, col_name, has_data_index, cur_node):
    col_index = col_dict[col_name]
    right_observe_index = has_data_index[col_index]
    observe_index = right_observe_index
    if len(cur_node) == 0:
        return False
    for node in cur_node:
        left_observe_index = has_data_index[node]
        observe_index = list(set(left_observe_index).intersection(set(observe_index)))
    if len(observe_index) == 0:
        return False
    newShang = 0
    label_data = encoder_data[col_index][observe_index]
    feature_data = []
    for node in cur_node:
        feature_data.append(encoder_data[node][observe_index])
    feature_data = np.transpose(np.array(feature_data))
    unique_values, counts = np.unique(feature_data, axis=0, return_counts=True)
    probabilities = counts / len(feature_data)
    for value, probability in zip(unique_values, probabilities):
        a = np.where(np.all(feature_data == value, axis=1))
        subset_labels = label_data[np.where(np.all(feature_data == value, axis=1))[0]]
        newShang += probability * entropy(subset_labels)
        if newShang > 0:
            return False
    if newShang == 0:
        return True
    else:
        return False


def train_new_CFD_model(new_X_index_list, y_index, un_satisfy_tuples, FD, new_x_code, fields, device):
    no, dim = new_x_code.shape
    all_index = set([i for i in range(no)])
    un_satisfy_tuples_index = set()
    for sub_set in un_satisfy_tuples:
        un_satisfy_tuples_index.update(sub_set)
    length = len(list(all_index - un_satisfy_tuples_index))
    set_index = list(all_index - un_satisfy_tuples_index)
    X = []
    begin_list = [0]
    begin = 0
    for index, field in enumerate(fields):
        if field.data_type == "Categorical Data":
            begin += len(field.dict)
        else:
            begin += 1
        begin_list.append(begin)
    for x in range(len(fields)):
        if x in new_X_index_list:
            cur_data = new_x_code[set_index]
            cur_data = cur_data[:, begin_list[x]:begin_list[x + 1]]
            X.append(cur_data)
        elif x != y_index:
            cur_data = torch.zeros((length, begin_list[x + 1] - begin_list[x])).to(device)
            X.append(cur_data)
    X = torch.cat(X, dim=1).to(device)
    Y = new_x_code[set_index]
    Y = Y[:, begin_list[y_index]:begin_list[y_index + 1]]
    Y = torch.argmax(Y, dim=1).long().to(device)
    input_dim = X.shape[1]
    output_dim = begin_list[y_index + 1] - begin_list[y_index]

    model = train_Model(X, Y, FD)
    return model


def get_trueProObserve(generate_x,decoder_z_impute,fields,data_m,device):
    truePro = torch.zeros(generate_x.shape[0], len(fields)).to(device)
    cur_index = 0
    for index, field in enumerate(fields):
        if field.data_type == "Categorical Data":
            dim = field.dim()
            data = generate_x[:, cur_index:cur_index + dim]
            zero_data = decoder_z_impute[:, cur_index:cur_index + dim]
            _, max_data_index = torch.max(data, dim=1, keepdim=True)
            _, max_zero_data_index = torch.max(zero_data, dim=1, keepdim=True)
            truePro[:, index] = torch.where(max_data_index == max_zero_data_index, torch.tensor(1).to(device),torch.tensor(0).to(device)).squeeze(-1)
            cur_index = cur_index + dim
        else:
            cur_index = cur_index + 1
    truePro = truePro.cpu().numpy()
    truePro = truePro * data_m
    return truePro



def get_eq_dict(values, miss_data_x, data_m):
    all_val_eq_dict = {}
    for attr_index,attr_name in enumerate(values):
        cur_dict = {}
        attr_data = miss_data_x.iloc[:,attr_index]
        for tup_index, val in enumerate(attr_data):
            if data_m[tup_index, attr_index] == 1:
                if val not in cur_dict.keys():
                    cur_dict[val] = [tup_index]
                else:
                    cur_dict[val].append(tup_index)
        all_val_eq_dict[attr_index] = cur_dict
    return all_val_eq_dict


def di_gui_get_sublist(split_subset, new_x_index_list, new_eq_dict, flag, impute_data):
    if flag == len(new_x_index_list):
        return
    new_split_sublist = []
    for sublist_begin in split_subset:
        if len(sublist_begin) == 1:
            continue
        cur_split_sublist = []
        other_attr = new_x_index_list[flag]
        other_attr_dict = new_eq_dict[other_attr]
        for element in sublist_begin:
            if not any(element in split for split in cur_split_sublist):
                other_attr_val = impute_data.iloc[element, other_attr]
                other_attr_sublist = other_attr_dict[other_attr_val]
                intersection_element = list(set(sublist_begin).intersection(set(other_attr_sublist)))
                new_split_sublist.append(intersection_element)
                cur_split_sublist.append(intersection_element)
    flag = flag + 1
    di_gui_get_sublist(new_split_sublist, new_x_index_list, new_eq_dict, flag, impute_data)
    return new_split_sublist


def get_satisfy_unsatisfy_tuples(RES_X, new_eq_dict, y_index, impute_data, tuple_acc_list):
    satisfy_tuples = []
    un_satisfy_tuples = []
    for sublist in RES_X:
        if len(sublist) == 1:
            continue
        y_attr_dict = new_eq_dict[y_index]
        other_attr_val = impute_data.iloc[sublist[0], y_index]
        other_attr_sublist = y_attr_dict[other_attr_val]
        intersection_element = list(set(sublist).intersection(set(other_attr_sublist)))
        if len(intersection_element) == len(sublist):
            satisfy_tuples.append(intersection_element)
            un_satisfy_tuples.append([])
        else:
            split_tuples = []
            for element in sublist:
                if not any(element in split for split in split_tuples):
                    y_attr_val = impute_data.iloc[element, y_index]
                    y_attr_sublist = y_attr_dict[y_attr_val]
                    intersection_element = list(set(sublist).intersection(set(y_attr_sublist)))
                    split_tuples.append(intersection_element)

            score_true_tup = []
            score_true_max = 0
            max_tup_score = 0
            for each_split_tuples in split_tuples:
                cur_score = 0
                cur_max_score = 0
                for tuple in each_split_tuples:
                    cur_score = cur_score + tuple_acc_list[tuple]
                    if tuple_acc_list[tuple] > cur_max_score:
                        cur_max_score = tuple_acc_list[tuple]
                if cur_max_score > max_tup_score:
                    max_tup_score = cur_max_score
                    score_true_tup = each_split_tuples
                
            satisfy_tuples.append(score_true_tup)
            un_satisfy_tuples.append(list(set(sublist) - set(score_true_tup)))
    return satisfy_tuples, un_satisfy_tuples

def flatten_tuple_groups(tuple_groups):
    rows = []
    for group in tuple_groups:
        rows.extend(group)
    return np.array(sorted(set(rows)), dtype=int)



def get_cfd_domain_rows(impute_data, x_index_list, y_index, x_pattern, y_pattern):
    """
    Get the domain according to the complete CFD pattern:
    - constant positions on X must be satisfied;
    - if y_pattern is not '_', Y must also satisfy that constant.
    """
    if len(impute_data) == 0:
        return np.array([], dtype=int)

    mask = np.ones(len(impute_data), dtype=bool)

    # 1. Filter by the lhs pattern first.
    for idx, pattern_val in zip(x_index_list, x_pattern):
        if pattern_val != '_':
            mask &= (impute_data.iloc[:, idx].astype(str).values == str(pattern_val))

    # 2. Then filter by the rhs pattern.
    if y_pattern != '_':
        mask &= (impute_data.iloc[:, y_index].astype(str).values == str(y_pattern))

    return np.where(mask)[0].astype(int)



def get_cfd_tuple_acc_list(cell_acc, x_index_list, y_index):
    index_list = list(x_index_list) + [y_index]
    if len(index_list) == 0:
        return np.ones(cell_acc.shape[0])
    return np.min(cell_acc[:, index_list], axis=1)


def get_cfd_satisfy_unsatisfy_tuples(domain_rows, x_index_list, y_index, impute_data, tuple_acc_list, y_pattern):
    satisfy_tuples = []
    un_satisfy_tuples = []

    if len(domain_rows) == 0:
        return satisfy_tuples, un_satisfy_tuples

    # Group by the complete lhs within the domain.
    eq_groups = defaultdict(list)
    for row_idx in domain_rows:
        x_val = tuple(str(impute_data.iloc[row_idx, x_idx]) for x_idx in x_index_list)
        eq_groups[x_val].append(int(row_idx))

    for group_rows in eq_groups.values():
        if len(group_rows) == 0:
            continue

        # ----------------------------------------------------------
        # Variable RHS: reuse the FD-style local group scoring logic within the current domain.
        # ----------------------------------------------------------
        if y_pattern == '_':
            if len(group_rows) == 1:
                continue

            y_groups = defaultdict(list)
            for row_idx in group_rows:
                y_val = str(impute_data.iloc[row_idx, y_index])
                y_groups[y_val].append(int(row_idx))

            # If this lhs equivalence class has only one rhs value, all rows satisfy the rule.
            if len(y_groups) == 1:
                satisfy_tuples.append(list(group_rows))
                un_satisfy_tuples.append([])
            else:
                # Select the strongest rhs subgroup as satisfy, and mark the others as unsatisfy.
                best_rows = []
                best_max_score = -1
                for cur_rows in y_groups.values():
                    cur_max_score = max(tuple_acc_list[row] for row in cur_rows)
                    if cur_max_score > best_max_score:
                        best_max_score = cur_max_score
                        best_rows = list(cur_rows)

                satisfy_tuples.append(best_rows)
                un_satisfy_tuples.append(sorted(list(set(group_rows) - set(best_rows))))

        # ----------------------------------------------------------
        # Constant RHS: since the domain has already been filtered by the complete pattern,
        # these rows in the domain all belong to the support samples of this CFD.
        # ----------------------------------------------------------
        else:
            satisfy_tuples.append(list(group_rows))
            un_satisfy_tuples.append([])

    return satisfy_tuples, un_satisfy_tuples




def get_CFD_score(satisfy_tuples, un_satisfy_tuples, tuple_acc_list):
    satisfy_score = 0
    for each_satisfy_tuples in satisfy_tuples:
        for each_tuple in each_satisfy_tuples:
            satisfy_score += tuple_acc_list[each_tuple]

    un_satisfy_score = 0
    max_un_sa_tup_score = 0
    all_satisfy_tup_inS = []
    all_satisfy_tup_inS_acc = []
    un_satisfy_tup = ''
    for un_satisfy_tuples_index, each_un_satisfy_tuples in enumerate(un_satisfy_tuples):
        for each_tuple in each_un_satisfy_tuples:
            un_satisfy_score += tuple_acc_list[each_tuple]
            if tuple_acc_list[each_tuple] > max_un_sa_tup_score:
                all_satisfy_tup_inS = []
                all_satisfy_tup_inS_acc = []
                un_satisfy_tup = ''
                max_un_sa_tup_score = tuple_acc_list[each_tuple]
                un_satisfy_tup = each_tuple
                all_satisfy_tup_inS = satisfy_tuples[un_satisfy_tuples_index]
                for tup in all_satisfy_tup_inS:
                    all_satisfy_tup_inS_acc.append(tuple_acc_list[tup])
    if satisfy_score + un_satisfy_score == 0:
        score = 0
    else:
        score = satisfy_score / (satisfy_score + un_satisfy_score)
    return score, max_un_sa_tup_score, un_satisfy_tup, all_satisfy_tup_inS, all_satisfy_tup_inS_acc

def is_x_pattern_subsumed(general_pattern, specific_pattern):
    """
    Check whether specific_pattern is covered by general_pattern.
    general = ['_']
    specific = ['wife']
    Returns True.
    """
    for g, s in zip(general_pattern, specific_pattern):
        if g == "_":
            continue
        if g != s:
            return False
    return True


def prune_redundant_cfd_candidates(candidates):
    """
    Delete the more specific rule whenever its x_pattern is covered by a more general rule.
    y_pattern is no longer required to be the same.
    """
    pruned = []

    for i, cand_i in enumerate(candidates):
        redundant = False
        for j, cand_j in enumerate(candidates):
            if i == j:
                continue

            if cand_i["x_index_list"] != cand_j["x_index_list"]:
                continue
            if cand_i["y_index"] != cand_j["y_index"]:
                continue

            # cand_j is more general and not exactly the same.
            if is_x_pattern_subsumed(cand_j["x_pattern"], cand_i["x_pattern"]):
                if cand_j["x_pattern"] != cand_i["x_pattern"]:
                    redundant = True
                    break

        if not redundant:
            pruned.append(cand_i)

    return pruned


def evaluate_CFD_candidate(x_index_list, y_index, x_pattern, y_pattern, impute_data, cell_acc):
    """
    Evaluate a CFD under the complete pattern domain:
    1. Get the domain first;
    2. Check whether violations exist in this domain;
    3. If there is no violation, directly return the keep status;
    4. If violations exist, reuse FD-style computation for sat / unsat / score / max_unsat.
    """
    domain_rows = get_cfd_domain_rows(
        impute_data=impute_data,
        x_index_list=x_index_list,
        y_index=y_index,
        x_pattern=x_pattern,
        y_pattern=y_pattern
    )

    tuple_acc_list = get_cfd_tuple_acc_list(cell_acc, x_index_list, y_index)

    has_violation = has_cfd_violation_in_pattern_domain(
        domain_rows=domain_rows,
        x_index_list=x_index_list,
        y_index=y_index,
        impute_data=impute_data,
        y_pattern=y_pattern
    )

    # ------------------------------------------------------------------
    # No violation: directly regard this CFD as valid in the current pattern domain.
    # ------------------------------------------------------------------
    if not has_violation:
        support = len(domain_rows)
        return {
            'score': 1.0,
            'max_unsat_score': 0.0,
            'worst_unsat_tuple': '',
            'all_satisfy_tup_inS': list(domain_rows),
            'all_satisfy_tup_inS_acc': [tuple_acc_list[row] for row in domain_rows],
            'support': support,
            'domain_rows': np.array(domain_rows, dtype=int),
            'satisfy_tuples': [list(domain_rows)] if len(domain_rows) > 0 else [],
            'un_satisfy_tuples': [[]] if len(domain_rows) > 0 else [],
            'satisfy_rows': list(domain_rows),
            'unsatisfy_rows': [],
            'tuple_acc_list': tuple_acc_list,
            'has_violation': False,
        }

    # ------------------------------------------------------------------
    # Violation exists: reuse FD-style scoring.
    # ------------------------------------------------------------------
    satisfy_tuples, un_satisfy_tuples = get_cfd_satisfy_unsatisfy_tuples(
        domain_rows=domain_rows,
        x_index_list=x_index_list,
        y_index=y_index,
        impute_data=impute_data,
        tuple_acc_list=tuple_acc_list,
        y_pattern=y_pattern
    )

    score, max_unsat_score, worst_unsat_tuple, all_satisfy_tup_inS, all_satisfy_tup_inS_acc = get_CFD_score(
        satisfy_tuples, un_satisfy_tuples, tuple_acc_list
    )

    satisfy_rows = flatten_tuple_groups(satisfy_tuples)
    unsatisfy_rows = flatten_tuple_groups(un_satisfy_tuples)
    # support = len(satisfy_rows) + len(unsatisfy_rows)
    support = len(satisfy_rows)

    return {
        'score': score,
        'max_unsat_score': max_unsat_score,
        'worst_unsat_tuple': worst_unsat_tuple,
        'all_satisfy_tup_inS': all_satisfy_tup_inS,
        'all_satisfy_tup_inS_acc': all_satisfy_tup_inS_acc,
        'support': support,
        'domain_rows': np.array(domain_rows, dtype=int),
        'satisfy_tuples': satisfy_tuples,
        'un_satisfy_tuples': un_satisfy_tuples,
        'satisfy_rows': satisfy_rows,
        'unsatisfy_rows': unsatisfy_rows,
        'tuple_acc_list': tuple_acc_list,
        'has_violation': True,
    }



def evaluate_CFD_model(model, impute_data, cell_acc):
    return evaluate_CFD_candidate(
        model.x_index_list,
        model.y_index,
        model.x_pattern,
        model.y_pattern,
        impute_data,
        cell_acc
    )


def is_valid_cfd_candidate(eval_result, min_support=100):
    return (
        eval_result['support'] >= min_support
        and (eval_result['score'] - eval_result['max_unsat_score']) > 0
    )


def candidate_from_eval(x_index_list, y_index, x_pattern, y_pattern, eval_result, impute_data,
                        cfd_type='refined', lhs_expand_pool=None, refine_depth=0):
    x_names = [impute_data.columns[idx] for idx in x_index_list]
    y_name = impute_data.columns[y_index]
    train_rows = eval_result['satisfy_rows']
    if len(train_rows) == 0:
        train_rows = eval_result['domain_rows']
    return {
        'x_index_list': list(x_index_list),
        'y_index': y_index,
        'x_pattern': list(x_pattern),
        'y_pattern': str(y_pattern),
        'domain_rows': np.array(eval_result['domain_rows'], dtype=int),
        'condition_index': np.array(train_rows, dtype=int),
        'cfd_type': cfd_type,
        'support': int(eval_result['support']),
        'conf': float(eval_result['score']),
        'score': float(eval_result['score']),
        'max_unsat_score': float(eval_result['max_unsat_score']),
        'lhs_expand_pool': [] if lhs_expand_pool is None else list(lhs_expand_pool),
        'refine_depth': int(refine_depth),
        'cfd_rule': format_cfd_rule(x_names, y_name, x_pattern, y_pattern),
    }



def deduplicate_candidate_dicts(candidates):
    new_candidates = []
    seen = set()
    for candidate in candidates:
        if candidate['cfd_rule'] not in seen:
            seen.add(candidate['cfd_rule'])
            new_candidates.append(candidate)
    return new_candidates



def select_candidates(candidates):
    candidates = deduplicate_candidate_dicts(candidates)
    candidates.sort(
        key=lambda item: (
            # item.get('score', 0.0) - item.get('max_unsat_score', 0.0),
            item.get('score', 0.0),
            item.get('support', 0),
        ),
        reverse=True,
    )
    return candidates


def _build_seed_CFD_model_from_structure(x_index_list, y_index, columns,
                                         lhs_expand_pool, max_lhs_size,
                                         min_support, refine_depth=0):
    class SeedCFD(object):
        pass

    seed = SeedCFD()
    seed.rule_kind = 'CFD'
    seed.x_index_list = list(x_index_list)
    seed.y_index = y_index
    seed.x_pattern = ['_'] * len(x_index_list)
    seed.y_pattern = '_'
    seed.lhs_expand_pool = list(lhs_expand_pool)
    seed.max_lhs_size = int(max_lhs_size)
    seed.min_support = int(min_support)
    seed.refine_depth = int(refine_depth)
    x_names = [columns[idx] for idx in x_index_list]
    y_name = columns[y_index]
    seed.cfd_rule = format_cfd_rule(x_names, y_name, seed.x_pattern, seed.y_pattern)
    return seed

def has_enough_repeated_support_for_lhs(x_index_list, impute_data,
                                        min_group_support=5,
                                        rows=None):
    if rows is None:
        rows = np.arange(len(impute_data), dtype=int)
    else:
        rows = np.array(rows, dtype=int)

    if len(rows) == 0:
        return False

    group_dict = defaultdict(list)
    for row_idx in rows:
        x_val = tuple(str(impute_data.iloc[row_idx, x_idx]) for x_idx in x_index_list)
        group_dict[x_val].append(int(row_idx))

    repeated_group_num = 0
    repeated_sample_num = 0
    for _, row_list in group_dict.items():
        if len(row_list) >= min_group_support:
            repeated_group_num += 1
            repeated_sample_num += len(row_list)

    return True


def build_refined_candidates_for_pattern(x_index_list, y_index, x_pattern, impute_data, cell_acc,
                                         min_support, lhs_expand_pool=None,
                                         cfd_type='refined', refine_depth=0,
                                         prefer_y_pattern=None, data_m=None):
    candidates = []

    # First obtain a base domain using only the lhs pattern for collecting / pruning y candidates.
    base_domain_rows = get_cfd_domain_rows(
        impute_data=impute_data,
        x_index_list=x_index_list,
        y_index=y_index,
        x_pattern=x_pattern,
        y_pattern='_'
    )

    if len(base_domain_rows) < min_support:
        return candidates

    y_variants = []

    # 1. Try prefer_y_pattern first.
    if prefer_y_pattern is not None:
        prefer_y_pattern = str(prefer_y_pattern)

        if prefer_y_pattern == '_':
            y_variants.append('_')
        else:
            # If data_m is available, first use the support upper bound to decide whether prefer_y_pattern is worth trying.
            if data_m is None:
                y_variants.append(prefer_y_pattern)
            else:
                prefer_ub = get_support_upper_bound_for_value(
                    impute_data=impute_data,
                    data_m=data_m,
                    attr_idx=y_index,
                    domain_rows=base_domain_rows,
                    value=prefer_y_pattern
                )
                if prefer_ub['support_upper_bound'] >= min_support:
                    y_variants.append(prefer_y_pattern)

    # 2. Variable RHS can always be tried.
    if '_' not in y_variants:
        y_variants.append('_')

    # 3. Constant RHS candidates.
    if data_m is None:
        # Backward compatibility: when data_m is unavailable, fall back to frequency-based enumeration in the current impute_data.
        y_counter = Counter(impute_data.iloc[base_domain_rows, y_index].astype(str).values)
        for y_val, _ in y_counter.most_common():
            y_val = str(y_val)
            if y_val not in y_variants:
                y_variants.append(y_val)
    else:
        # New logic: sort by support upper bound and stop once it falls below min_support.
        y_value_items = get_candidate_values_by_support_upper_bound(
            impute_data=impute_data,
            data_m=data_m,
            attr_idx=y_index,
            domain_rows=base_domain_rows,
            min_support=min_support,
            exclude_value=None
        )

        for item in y_value_items:
            y_val = str(item['value'])
            if y_val not in y_variants:
                y_variants.append(y_val)

    for y_pattern in y_variants:
        eval_result = evaluate_CFD_candidate(
            x_index_list=x_index_list,
            y_index=y_index,
            x_pattern=x_pattern,
            y_pattern=y_pattern,
            impute_data=impute_data,
            cell_acc=cell_acc
        )

        if is_valid_cfd_candidate(eval_result, min_support=min_support):
            candidates.append(candidate_from_eval(
                x_index_list=x_index_list,
                y_index=y_index,
                x_pattern=x_pattern,
                y_pattern=y_pattern,
                eval_result=eval_result,
                impute_data=impute_data,
                cfd_type=cfd_type,
                lhs_expand_pool=lhs_expand_pool,
                refine_depth=refine_depth,
            ))

    return deduplicate_candidate_dicts(candidates)

def _as_numpy_data_m(data_m):
    """
    Convert data_m to a numpy array.
    Convention:
    - data_m[row, col] == 1 indicates an originally observed value;
    - data_m[row, col] == 0 indicates an originally missing value.
    """
    if isinstance(data_m, torch.Tensor):
        return data_m.detach().cpu().numpy()
    return np.asarray(data_m)


def get_table_missing_row_count(data_m, cols=None):
    """
    Backward-compatible helper: count table-level rows containing missing values.

    The current CFD mining threshold no longer uses this table-level count; it actually uses
    get_cfd_x_mis_count(...) to compute X_mis separately for each candidate CFD.
    """
    data_m_np = _as_numpy_data_m(data_m)
    if data_m_np.size == 0:
        return 0
    if cols is not None:
        cols = list(cols)
        if len(cols) == 0:
            return 0
        data_m_np = data_m_np[:, cols]
    return int(np.sum(np.any(data_m_np == 0, axis=1)))



def get_cfd_x_mis_count(miss_data, data_m, candidate):
    """
    Compute the X_mis count of a candidate CFD.

    For candidate CFD phi': (Z' -> A, t_p'):
    X_mis is the number of tuples that contain at least one originally missing value on Z' ∪ {A}
    and may still match t_p' according to the currently observed values.

    Matching rules:
    - If a pattern position is constant and the tuple is observed on that attribute but not equal to
      the constant, then it cannot match;
    - If a pattern position is constant and the tuple is missing on that attribute, it may match after imputation;
    - If a pattern position is '_', the value is unrestricted;
    - A tuple is counted in X_mis only when Z' ∪ {A} contains at least one missing value.
    """
    data_m_np = _as_numpy_data_m(data_m)
    if len(miss_data) == 0 or data_m_np.size == 0:
        return 0

    x_index_list = list(candidate.get('x_index_list', []))
    y_index = int(candidate.get('y_index'))
    x_pattern = list(candidate.get('x_pattern', ['_'] * len(x_index_list)))
    y_pattern = str(candidate.get('y_pattern', '_'))

    involved_cols = list(x_index_list) + [y_index]
    if len(involved_cols) == 0:
        return 0

    count = 0
    for row_idx in range(len(miss_data)):
        # There must be at least one original missing value on Z' ∪ {A}.
        has_missing = False
        for col_idx in involved_cols:
            if data_m_np[row_idx, col_idx] == 0:
                has_missing = True
                break
        if not has_missing:
            continue

        may_match = True

        # Check constant positions in the LHS pattern.
        for pos, attr_idx in enumerate(x_index_list):
            if pos >= len(x_pattern):
                continue
            pattern_val = str(x_pattern[pos])
            if pattern_val == '_':
                continue
            if data_m_np[row_idx, attr_idx] == 1:
                cur_val = str(miss_data.iloc[row_idx, attr_idx])
                if cur_val != pattern_val:
                    may_match = False
                    break

        if not may_match:
            continue

        # Check constant positions in the RHS pattern.
        if y_pattern != '_' and data_m_np[row_idx, y_index] == 1:
            cur_y_val = str(miss_data.iloc[row_idx, y_index])
            if cur_y_val != y_pattern:
                may_match = False

        if may_match:
            count += 1

    return int(count)



def get_cfd_missing_possible_count(miss_data, data_m, candidate):
    """Backward-compatible old function name: actually calls get_cfd_x_mis_count."""
    return get_cfd_x_mis_count(miss_data, data_m, candidate)

def get_cfd_candidate_pool_from_model(model):
    """
    Read the candidate pool generated and fixed during the CFD mining stage.
    Prefer cfd_candidate_pool on the model; if unavailable, fall back to the global candidate pool.
    """
    pool = getattr(model, 'cfd_candidate_pool', None)
    if pool is None:
        pool = GLOBAL_CFD_CANDIDATE_POOL
    return [] if pool is None else list(pool)


def is_same_lhs_rhs_candidate(candidate, model):
    """Check whether the candidate CFD has the same lhs -> rhs structure as the current problematic CFD."""
    return (
        int(candidate.get('y_index')) == int(model.y_index)
        and list(candidate.get('x_index_list', [])) == list(model.x_index_list)
    )


def is_lhs_expanded_candidate(candidate, model, target_lhs_size, continuous_cols=None):
    """
    Check whether the candidate CFD is obtained by adding attributes on top of the current
    problematic CFD lhs -> rhs.

    Constraints:
    1. The RHS attribute must be the same;
    2. The LHS of the current problematic CFD must be a subset of the candidate LHS;
    3. The number of attributes in the candidate LHS must equal target_lhs_size;
    4. Newly added attributes cannot be continuous attributes;
    5. The number of candidate LHS attributes cannot exceed max_lhs_size.
    """
    if int(candidate.get('y_index')) != int(model.y_index):
        return False

    old_lhs = list(model.x_index_list)
    new_lhs = list(candidate.get('x_index_list', []))

    max_lhs_size = int(getattr(model, 'max_lhs_size', 3))
    target_lhs_size = int(target_lhs_size)

    if target_lhs_size <= len(old_lhs):
        return False
    if target_lhs_size > max_lhs_size:
        return False
    if len(new_lhs) != target_lhs_size:
        return False
    if not set(old_lhs).issubset(set(new_lhs)):
        return False

    added_attrs = [idx for idx in new_lhs if idx not in old_lhs]
    if len(added_attrs) != target_lhs_size - len(old_lhs):
        return False

    continuous_cols = set([] if continuous_cols is None else list(continuous_cols))
    for attr_idx in added_attrs:
        if attr_idx in continuous_cols:
            return False

    return True


def is_lhs_plus_one_candidate(candidate, model, continuous_cols=None):
    """Backward-compatible old call: check whether the candidate CFD only adds one attribute to the current LHS."""
    return is_lhs_expanded_candidate(
        candidate=candidate,
        model=model,
        target_lhs_size=len(model.x_index_list) + 1,
        continuous_cols=continuous_cols
    )


def get_support_upper_bound_for_value(impute_data, data_m, attr_idx, domain_rows, value):
    """
    Compute the support upper bound for a candidate value:

    support_upper_bound(value)
    = number of rows in the domain where attr_idx is observed and equal to value
      + number of rows in the domain where attr_idx is originally missing

    Notes:
    - Missingness is determined by data_m;
    - value comparisons are uniformly converted to str.
    """
    data_m_np = _as_numpy_data_m(data_m)
    domain_rows = np.array(domain_rows, dtype=int)

    if len(domain_rows) == 0:
        return {
            'value': str(value),
            'support_upper_bound': 0,
            'observed_equal_count': 0,
            'missing_count': 0,
        }

    attr_observed_mask = data_m_np[domain_rows, attr_idx] == 1
    attr_missing_mask = data_m_np[domain_rows, attr_idx] == 0

    observed_rows = domain_rows[attr_observed_mask]
    missing_count = int(np.sum(attr_missing_mask))

    if len(observed_rows) == 0:
        observed_equal_count = 0
    else:
        observed_values = impute_data.iloc[observed_rows, attr_idx].astype(str).values
        observed_equal_count = int(np.sum(observed_values == str(value)))

    support_upper_bound = observed_equal_count + missing_count

    return {
        'value': str(value),
        'support_upper_bound': int(support_upper_bound),
        'observed_equal_count': int(observed_equal_count),
        'missing_count': int(missing_count),
    }


def get_candidate_values_by_support_upper_bound(impute_data, data_m, attr_idx,
                                                domain_rows, min_support,
                                                exclude_value=None):
    """
    Return the candidate value list sorted by support upper bound in descending order.

    Enumeration strategy:
    1. Candidate values come from all values that have appeared in this attribute column of the current impute_data;
    2. For each candidate value v, compute:
       support_upper_bound(v)
       = number of rows in the domain where attr is observed and equal to v
         + number of rows in the domain where attr is originally missing
    3. Sort by support_upper_bound in descending order;
    4. Stop enumeration once support_upper_bound < min_support.

    Returned item format:
    {
        'value': str,
        'support_upper_bound': int,
        'observed_equal_count': int,
        'missing_count': int,
        'global_count': int,
    }
    """
    domain_rows = np.array(domain_rows, dtype=int)
    min_support = int(min_support)

    if len(domain_rows) == 0:
        return []

    # Candidate value set: all values that appear in the current column.
    all_series = impute_data.iloc[:, attr_idx].dropna().astype(str)
    candidate_values = all_series.unique().tolist()

    if len(candidate_values) == 0:
        return []

    # Use the current full-column frequency as a tie-breaker, not as the primary sorting criterion.
    global_counter = Counter(all_series.values)

    value_items = []
    for value in candidate_values:
        value = str(value)

        if exclude_value is not None and value == str(exclude_value):
            continue

        ub_info = get_support_upper_bound_for_value(
            impute_data=impute_data,
            data_m=data_m,
            attr_idx=attr_idx,
            domain_rows=domain_rows,
            value=value
        )

        ub_info['global_count'] = int(global_counter.get(value, 0))
        value_items.append(ub_info)

    value_items.sort(
        key=lambda item: (
            item['support_upper_bound'],
            item['observed_equal_count'],
            item['global_count']
        ),
        reverse=True
    )

    kept_items = []
    for item in value_items:
        if item['support_upper_bound'] < min_support:
            break
        kept_items.append(item)

    return kept_items


def get_lhs_relaxed_domain_rows_for_value_refinement(impute_data, x_index_list, y_index,
                                                     x_pattern, relax_pos=None):
    """
    Compute the domain used to try a candidate value at a certain lhs pattern position.

    For lhs specialization:
    - The current position is originally '_';
    - relax_pos may be specified for this position or omitted;
    - The domain is equivalent to the current lhs pattern domain.

    For lhs replacement:
    - The current position is a constant value;
    - This position must be temporarily relaxed to '_' first;
    - Otherwise, candidate new values would be incorrectly counted only within the old value domain.

    y_pattern is fixed as '_' here because:
    - This function is only responsible for computing the support upper bound for lhs value candidates;
    - build_refined_candidates_for_pattern will later continue to try y_pattern;
    - If filtering is done here by the old constant y_pattern in advance, y_pattern='_' or other constant RHS candidates may be pruned incorrectly.
    """
    relaxed_x_pattern = list(x_pattern)

    if relax_pos is not None:
        relaxed_x_pattern[relax_pos] = '_'

    return get_cfd_domain_rows(
        impute_data=impute_data,
        x_index_list=x_index_list,
        y_index=y_index,
        x_pattern=relaxed_x_pattern,
        y_pattern='_'
    )

def build_support_upper_bound_cfd_pool(x_index_list, y_index, impute_data, data_m,
                                       min_support, seed_x_pattern=None,
                                       prefer_y_pattern=None,
                                       lhs_expand_pool=None,
                                       refine_depth=0,
                                       cfd_type='support_upper_bound_pool'):
    """
    Build a CFD candidate pool that satisfies support_upper_bound.

    This pool is only responsible for support upper bound filtering; this function does not judge
    whether a CFD truly holds. validate_support_upper_bound_cfd_pool(...) later validates each CFD
    in the pool one by one.

    For a fixed lhs -> rhs structure:
    1. Enumerate constant/_ patterns that actually appear on lhs;
    2. Keep y_pattern='_' first for each lhs pattern;
    3. Use get_candidate_values_by_support_upper_bound(...) to enumerate constant RHS values that satisfy the RHS support upper bound;
    4. Return a unified candidate dict format for later model training.
    """
    min_support = int(min_support)
    x_index_list = list(x_index_list)
    candidates = []
    seen_rule = set()

    if len(x_index_list) == 0 or len(impute_data) == 0:
        return candidates

    x_names = [impute_data.columns[idx] for idx in x_index_list]
    y_name = impute_data.columns[y_index]

    # ----------------------------------------------------------
    # 1. Enumerate lhs patterns.
    #    Put seed_x_pattern into the pool first to ensure that the current problematic CFD pattern is considered first.
    # ----------------------------------------------------------
    x_patterns = []
    seen_pattern = set()

    def _add_x_pattern(pattern):
        if pattern is None:
            return
        pattern = [str(v) for v in pattern]
        if len(pattern) != len(x_index_list):
            return
        key = tuple(pattern)
        if key not in seen_pattern:
            seen_pattern.add(key)
            x_patterns.append(pattern)

    _add_x_pattern(seed_x_pattern)
    _add_x_pattern(['_'] * len(x_index_list))

    for row_idx in range(len(impute_data)):
        x_values = [str(impute_data.iloc[row_idx, attr_idx]) for attr_idx in x_index_list]
        for pattern in generate_patterns_from_values(x_values):
            _add_x_pattern(pattern)

    # ----------------------------------------------------------
    # 2. Build RHS candidates for each lhs pattern.
    # ----------------------------------------------------------
    for x_pattern in x_patterns:
        base_domain_rows = get_cfd_domain_rows(
            impute_data=impute_data,
            x_index_list=x_index_list,
            y_index=y_index,
            x_pattern=x_pattern,
            y_pattern='_'
        )

        # For y_pattern='_', the support upper bound is the size of the lhs pattern domain.
        if len(base_domain_rows) < min_support:
            continue

        y_patterns = []

        def _add_y_pattern(y_pattern):
            y_pattern = str(y_pattern)
            if y_pattern not in y_patterns:
                y_patterns.append(y_pattern)

        if prefer_y_pattern is not None:
            prefer_y_pattern = str(prefer_y_pattern)
            if prefer_y_pattern == '_':
                _add_y_pattern('_')
            else:
                prefer_ub = get_support_upper_bound_for_value(
                    impute_data=impute_data,
                    data_m=data_m,
                    attr_idx=y_index,
                    domain_rows=base_domain_rows,
                    value=prefer_y_pattern
                )
                if prefer_ub['support_upper_bound'] >= min_support:
                    _add_y_pattern(prefer_y_pattern)

        _add_y_pattern('_')

        y_value_items = get_candidate_values_by_support_upper_bound(
            impute_data=impute_data,
            data_m=data_m,
            attr_idx=y_index,
            domain_rows=base_domain_rows,
            min_support=min_support,
            exclude_value=None
        )
        for item in y_value_items:
            _add_y_pattern(str(item['value']))

        for y_pattern in y_patterns:
            if y_pattern == '_':
                support_upper_bound = len(base_domain_rows)
            else:
                ub_info = get_support_upper_bound_for_value(
                    impute_data=impute_data,
                    data_m=data_m,
                    attr_idx=y_index,
                    domain_rows=base_domain_rows,
                    value=y_pattern
                )
                support_upper_bound = ub_info['support_upper_bound']
                if support_upper_bound < min_support:
                    continue

            cfd_rule = format_cfd_rule(x_names, y_name, x_pattern, y_pattern)
            if cfd_rule in seen_rule:
                continue
            seen_rule.add(cfd_rule)

            append_support_ub_cfd_rule(cfd_rule)
            candidates.append({
                'x_index_list': list(x_index_list),
                'y_index': y_index,
                'x_pattern': list(x_pattern),
                'y_pattern': str(y_pattern),
                'domain_rows': np.array(base_domain_rows, dtype=int),
                'condition_index': np.array([], dtype=int),
                'cfd_type': cfd_type,
                'support_upper_bound': int(support_upper_bound),
                'support': int(len(base_domain_rows)),
                'conf': 0.0,
                'lhs_expand_pool': [] if lhs_expand_pool is None else list(lhs_expand_pool),
                'refine_depth': int(refine_depth),
                'cfd_rule': cfd_rule,
            })

    return candidates


def validate_support_upper_bound_cfd_pool(pool_candidates, impute_data, cell_acc,
                                          min_support, cfd_type='support_ub_validated'):
    """
    Validate the support_upper_bound CFD pool one by one.
    Only CFDs that truly pass is_valid_cfd_candidate(...) enter the returned list.
    """
    valid_candidates = []

    for pool_cand in pool_candidates:

        eval_result = evaluate_CFD_candidate(
            x_index_list=pool_cand['x_index_list'],
            y_index=pool_cand['y_index'],
            x_pattern=pool_cand['x_pattern'],
            y_pattern=pool_cand['y_pattern'],
            impute_data=impute_data,
            cell_acc=cell_acc
        )

        if not is_valid_cfd_candidate(eval_result, min_support=min_support):
            continue

        candidate = candidate_from_eval(
            x_index_list=pool_cand['x_index_list'],
            y_index=pool_cand['y_index'],
            x_pattern=pool_cand['x_pattern'],
            y_pattern=pool_cand['y_pattern'],
            eval_result=eval_result,
            impute_data=impute_data,
            cfd_type=cfd_type,
            lhs_expand_pool=pool_cand.get('lhs_expand_pool', []),
            refine_depth=pool_cand.get('refine_depth', 0),
        )
        candidate['support_upper_bound'] = int(pool_cand.get('support_upper_bound', candidate['support']))
        valid_candidates.append(candidate)

    valid_candidates = deduplicate_candidate_dicts(valid_candidates)
    valid_candidates = prune_redundant_cfd_candidates(valid_candidates)
    valid_candidates = select_candidates(valid_candidates)
    return valid_candidates


def find_valid_cfds_with_same_lhs_rhs(model, impute_data, cell_acc, data_m):
    """
    Stage 1:
    No longer build the support_upper_bound CFD pool on the fly.
    Directly search the candidate pool saved during CFD mining for candidate CFDs with the same
    lhs -> rhs structure as the current problematic CFD, and validate them one by one.
    """
    min_support = int(getattr(model, 'min_support', GLOBAL_CFD_MIN_SUPPORT))
    pool_candidates = [
        cand for cand in get_cfd_candidate_pool_from_model(model)
        if is_same_lhs_rhs_candidate(cand, model)
    ]

    return validate_support_upper_bound_cfd_pool(
        pool_candidates=pool_candidates,
        impute_data=impute_data,
        cell_acc=cell_acc,
        min_support=min_support,
        cfd_type='same_lhs_rhs_validated_from_mined_pool'
    )


def find_valid_cfds_by_expanding_lhs_until_max(model, impute_data, cell_acc, data_m, continuous_cols):
    """
    Stage 2:
    If no valid CFD is found in the candidate pool with the same lhs -> rhs, continue searching
    the candidate pool saved during mining by expanding the number of LHS attributes layer by layer:

        |lhs| + 1, |lhs| + 2, ..., max_lhs_size

    At each layer, only search existing pool candidates that satisfy old_lhs \subset new_lhs and have
    the same RHS, then validate them one by one. Once valid CFDs are found at a layer, return all valid
    candidates from that layer and stop expanding to larger LHS sizes.
    """
    min_support = int(getattr(model, 'min_support', GLOBAL_CFD_MIN_SUPPORT))
    max_lhs_size = int(getattr(model, 'max_lhs_size', 3))
    cur_lhs_size = len(model.x_index_list)

    if cur_lhs_size >= max_lhs_size:
        return []

    mined_pool = get_cfd_candidate_pool_from_model(model)

    for target_lhs_size in range(cur_lhs_size + 1, max_lhs_size + 1):
        pool_candidates = [
            cand for cand in mined_pool
            if is_lhs_expanded_candidate(
                candidate=cand,
                model=model,
                target_lhs_size=target_lhs_size,
                continuous_cols=continuous_cols
            )
        ]

        if len(pool_candidates) == 0:
            continue

        valid_candidates = validate_support_upper_bound_cfd_pool(
            pool_candidates=pool_candidates,
            impute_data=impute_data,
            cell_acc=cell_acc,
            min_support=min_support,
            cfd_type=f'lhs_size_{target_lhs_size}_validated_from_mined_pool'
        )

        if len(valid_candidates) > 0:
            return valid_candidates

    return []


def find_valid_cfds_by_adding_one_lhs_attr(model, impute_data, cell_acc, data_m, continuous_cols):
    return find_valid_cfds_by_expanding_lhs_until_max(
        model=model,
        impute_data=impute_data,
        cell_acc=cell_acc,
        data_m=data_m,
        continuous_cols=continuous_cols
    )


def train_new_CFD_model(candidate_rule, new_x_code, fields, device):
    set_index = np.array(candidate_rule.get('condition_index', []), dtype=int)
    if len(set_index) == 0:
        return None
    X, begin_list = build_rule_input(new_x_code[set_index], candidate_rule['x_index_list'], candidate_rule['y_index'], fields)
    y_slice = new_x_code[set_index][:, begin_list[candidate_rule['y_index']]:begin_list[candidate_rule['y_index'] + 1]]
    if y_slice.shape[1] <= 1:
        return None
    Y = torch.argmax(y_slice, dim=1).long().to(device)
    input_dim = X.shape[1]
    output_dim = begin_list[candidate_rule['y_index'] + 1] - begin_list[candidate_rule['y_index']]
    model = CFDModel(input_dim, output_dim, candidate_rule['x_index_list'], candidate_rule['y_index']).to(device)
    model = train_Model(X.to(device), Y, model)
    return model

def has_cfd_violation_in_pattern_domain(domain_rows, x_index_list, y_index, impute_data, y_pattern):
    """
    Check whether the current CFD has violating tuples in the current pattern domain.

    Rules:
    1. If y_pattern is a constant:
       Since the domain has already been filtered by the complete pattern, there is no violation by default in the current domain.
    2. If y_pattern is '_':
       Group by lhs within the domain; if any lhs group contains multiple rhs values, it is considered a violation.
    """
    if len(domain_rows) == 0:
        return False

    # Constant RHS: by default, there is no violation under the complete pattern domain.
    if y_pattern != '_':
        return False

    eq_groups = defaultdict(list)
    for row_idx in domain_rows:
        x_val = tuple(str(impute_data.iloc[row_idx, x_idx]) for x_idx in x_index_list)
        eq_groups[x_val].append(int(row_idx))

    for group_rows in eq_groups.values():
        if len(group_rows) <= 1:
            continue

        y_values = set(str(impute_data.iloc[row_idx, y_index]) for row_idx in group_rows)
        if len(y_values) > 1:
            return True

    return False


def update_CFD_models(generate_x, zero_feed_data, fields, data_m, M_tensor,
                      value_cat, values, miss_data_x, enc, device,
                      CFD_model_list, cell_acc, continuous_cols, cost_time):
    """
    CFD update logic:

    1. First convert the current generated result back to impute_data;
    2. Evaluate each old CFD;
    3. If an old rule is valid, keep it directly;
    4. If an old rule is invalid:
       4.1 Do not build a candidate pool on the fly; instead, search the candidate pool already saved
           during CFD mining for candidate CFDs with the same lhs -> rhs and validate them one by one;
       4.2 If no valid CFD is found for the same lhs -> rhs, search the same candidate pool for candidates
           in the form attr + lhs -> rhs and validate them one by one;
    5. Train new models for validated candidates and keep them;
    6. Save updated rules to cfd_rules_updated.txt.
    """

    # ----------------------------------------------------------
    # 1. Generate the current imputation result and convert it back to a DataFrame.
    # ----------------------------------------------------------
    new_x_code = generate_x * (1 - M_tensor) + M_tensor * zero_feed_data
    impute_data = reconvert_data(new_x_code, fields, value_cat, values, miss_data_x, data_m, enc)
    impute_data = pd.DataFrame(impute_data, columns=values)

    new_CFD_model_list = []
    kept_rules = []
    seen_rule = set()

    # ----------------------------------------------------------
    # 2. Check old rules one by one.
    # ----------------------------------------------------------
    for model in CFD_model_list:
        cur_eval = evaluate_CFD_model(model, impute_data, cell_acc)
        min_support = int(getattr(model, 'min_support', 100))

        # ------------------------------------------------------
        # 3. Valid old rules: keep them directly.
        # ------------------------------------------------------
        if is_valid_cfd_candidate(cur_eval, min_support=min_support):
            if model.cfd_rule not in seen_rule:
                kept_rules.append(model.cfd_rule)
                new_CFD_model_list.append(model)
                seen_rule.add(model.cfd_rule)
            continue

        # ------------------------------------------------------
        # 4. Invalid old rules: first search the support_upper_bound CFD pool with the same lhs -> rhs.
        # ------------------------------------------------------
        # start = time.time()
        refined_candidates = find_valid_cfds_with_same_lhs_rhs(
            model=model,
            impute_data=impute_data,
            cell_acc=cell_acc,
            data_m=data_m
        )
        # end = time.time()
        # print(f"Total elapsed time: {end-start}")

        # ------------------------------------------------------
        # 5. If no valid CFD is found for the same lhs -> rhs, expand LHS attributes layer by layer until max_lhs_size.
        # ------------------------------------------------------
        if len(refined_candidates) == 0:
            refined_candidates = find_valid_cfds_by_adding_one_lhs_attr(
                model=model,
                impute_data=impute_data,
                cell_acc=cell_acc,
                data_m=data_m,
                continuous_cols=continuous_cols
            )

        # ------------------------------------------------------
        # 6. Deduplicate, prune, and sort.
        # ------------------------------------------------------
        refined_candidates = deduplicate_candidate_dicts(refined_candidates)
        for cand in refined_candidates:
            cand['x_index_list'] = list(cand['x_index_list'])
            cand['y_index'] = cand['y_index']

        refined_candidates = prune_redundant_cfd_candidates(refined_candidates)
        refined_candidates = select_candidates(refined_candidates)

        # ------------------------------------------------------
        # 7. Train and accept candidate rules.
        # ------------------------------------------------------
        for candidate in refined_candidates:
            if candidate['cfd_rule'] in seen_rule:
                continue

            start_time = time.time()
            new_model = train_new_CFD_model(candidate, new_x_code, fields, device)
            train_time = time.time() - start_time
            cost_time += train_time

            if new_model is None:
                continue

            attach_cfd_metadata(
                new_model,
                candidate,
                new_x_code,
                fields,
                min_support=min_support,
                max_lhs_size=getattr(model, 'max_lhs_size', 3),
                lhs_expand_pool=candidate.get('lhs_expand_pool', getattr(model, 'lhs_expand_pool', []))
            )
            new_model.cfd_candidate_pool = get_cfd_candidate_pool_from_model(model)
            new_model.min_observed_support = int(getattr(model, 'min_observed_support', GLOBAL_CFD_MIN_OBSERVED_SUPPORT))
            new_model.x_mis_count = int(candidate.get('x_mis_count', getattr(model, 'min_x_mis_count', GLOBAL_CFD_MIN_X_MIS_COUNT)))
            new_model.min_x_mis_count = int(getattr(model, 'min_x_mis_count', GLOBAL_CFD_MIN_X_MIS_COUNT))
            new_model.support_upper_bound = int(candidate.get('support_upper_bound', candidate.get('support', 0)))

            kept_rules.append(candidate['cfd_rule'])
            new_CFD_model_list.append(new_model)
            seen_rule.add(candidate['cfd_rule'])

    # ----------------------------------------------------------
    # 8. Final deduplication and saving.
    # ----------------------------------------------------------
    unique_rule_list = []
    seen_tmp = set()
    for rule in kept_rules:
        if rule not in seen_tmp:
            unique_rule_list.append(rule)
            seen_tmp.add(rule)

    save_cfd_rules(unique_rule_list, save_dir='out/CFD_list/', file_name='cfd_rules_updated_new.txt')

    return new_CFD_model_list
