from Levenshtein import distance as lev
import math
import pandas as pd
import numpy as np
from tqdm import tqdm

def compute_distance(a, b, dtype):
    if isinstance(a, str) and dtype == "int64":
        a=int(a)
    elif isinstance(a, str) and dtype == "float64":
        a=float(a)

    if isinstance(b, str) and dtype == "int64":
        b=int(b)
    elif isinstance(b, str) and dtype == "float64":
        b=float(b)

    if dtype == "int64" or dtype == "float64":
        if math.isnan(a) or math.isnan(b):
            return -1
        return round(abs(a - b), 2)
    else:
        if a == "?" or b == "?" or a == "nan" or b == "nan":
            return -1
        return lev(str(a), str(b))



def load_metadata_reduced(df: pd.DataFrame):
    ordered_distances = {}
    for col in df.columns:
        ordered_distances[col] = set()
        dtype = df[col].dtype
        uniques=df[col].unique()
        for i in tqdm(range(0, len(uniques)), desc=f"Computing distance for {col}"):
            for j in range(i, len(uniques)):
                ordered_distances[col].add(compute_distance(uniques[i], uniques[j], dtype))
        ordered_distances[col]=list(ordered_distances[col])
        ordered_distances[col].sort()
    return ordered_distances


def load_metadata(df: pd.DataFrame):
    distances_map = {}
    for col in df.columns:
        values = df[col].values
        distances_map[col] = {}
        dtype = df[col].dtype
        for i in range(len(values) - 1):
            for j in range(i + 1, len(values)):
                distance = compute_distance(values[i], values[j], dtype)
                if not distance in distances_map[col]:
                    distances_map[col][distance] = []
                distances_map[col][distance].append((values[i], values[j]))
    stats_map = {}
    for col in df.columns:
        stats_map[col] = {}
        stats_map[col]["type"] = df[col].dtype
        if df[col].dtype == "int64" or df[col].dtype == "float64":
            stats_map[col]["mean"] = df[col].mean()
            stats_map[col]["median"] = df[col].median()
            stats_map[col]["max"] = df[col].max()
            stats_map[col]["min"] = df[col].min()
        else:  # df[col].dtype=="object":
            values = list(map(lambda x: len(str(x)), df[col].values))
            stats_map[col]["max_len"] = max(values)
            stats_map[col]["min_len"] = min(values)
        stats_map[col]["mode"] = list(df[col].mode().values)
        stats_map[col]["distances_stats"] = {}
        count = 0
        stats_map[col]["distances_stats"]["mean"] = 0
        for k in distances_map[col]:
            stats_map[col]["distances_stats"]["mean"] += (k * len(distances_map[col][k]))
            count += len(distances_map[col][k])
        stats_map[col]["distances_stats"]["mean"] = stats_map[col]["distances_stats"]["mean"] / count
        stats_map[col]["distances_stats"]["max_distance"] = max(distances_map[col])
        stats_map[col]["distances_stats"]["min_distance"] = min(distances_map[col])
        stats_map[col]["ordered_distances"] = list(distances_map[col].keys())
        stats_map[col]["ordered_distances"].sort()
        if stats_map[col]["distances_stats"]["min_distance"] <= 0 and len(distances_map[col]) > 1:
            temp = list(distances_map[col])
            temp.sort()
            stats_map[col]["distances_stats"]["min_non_zero"] = temp[temp.index(0) + 1]
        stats_map[col]["distances_stats"]["variance"] = np.var(np.array(list(distances_map[col])))

    return distances_map, stats_map


def get_initial_threshold(distances_map, stats_map):
    to_ret = []
    for e in stats_map:
        if "min_non_zero" in stats_map[e]["distances_stats"]:
            thr = stats_map[e]["distances_stats"]["min_non_zero"]
        else:
            thr = stats_map[e]["distances_stats"]["min_distance"]
        valid_comparison_counter = 0
        for d in distances_map[e]:
            if d <= thr:
                valid_comparison_counter += len(distances_map[e][d])
        to_ret.append(thr)
    return list(map(lambda x:float(x),to_ret))