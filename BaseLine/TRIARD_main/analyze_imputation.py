from datetime import datetime

import pandas as pd
from dependencies_util import *

import dataset_util
import util
import re
import json
from load_metadata import compute_distance


def extract_dependency_from_json(json_file, attrs, key="violatingRFD"):
    lhs, rhs = json_file[key]["LHS"], json_file[key]["RHS"]
    rhs = [util.cast_attribute(f"{rhs['name']}@{rhs['threshold']}")]
    lhs = list(map(lambda x: util.cast_attribute(f"{x['name']}@{x['threshold']}"), lhs))
    try:

        return (AttributeSet(lhs), AttributeSet(rhs), attrs[json_file["fieldIndex"]]) if "fieldIndex" in json_file else (AttributeSet(lhs), AttributeSet(rhs))
    except:
        return []


def compute_score(attribute_result, count_impossible=True):
    imputation_result_count = {"missing": 0, "correct": 0, "incorrect": 0, "impossible": 0}
    for i, r in attribute_result.iterrows():
        if r["valore imputato"] == "?":
            imputation_result_count["missing"] += 1
        elif r["valore imputato"] == "!" and count_impossible:
            imputation_result_count["impossible"] += 1
        elif r["valore imputato"] == r["valore da imputare"]:
            imputation_result_count["correct"] += 1
        else:
            imputation_result_count["incorrect"] += 1
    return round((imputation_result_count["correct"]+imputation_result_count["impossible"]) / len(attribute_result), 10)


def get_used_dependencies(attribute_result, used_to_impute, only_exact=True):
    dependencies = set()
    for i, r in attribute_result.iterrows():
        if r["valore imputato"] == "?":
            continue
        extract = False
        if r["valore imputato"] == r["valore da imputare"] and only_exact:
            extract = True
        elif r["valore imputato"] != r["valore da imputare"] and only_exact == False:
            extract = True
        if extract:
            dependencies.add(used_to_impute[r["riga"]][r["nome attributo"]]["set"])
    return dependencies


def get_similar_row_by_rhs(df, lhs, rhs, index_row_to_impute, rhs_value_to_impute, threshold_map):
    filtered_df = df[[rhs]+lhs]
    row_to_impute=filtered_df.loc[index_row_to_impute]
    filtered_df=filtered_df.dropna()

    rows=[]
    for i, r in filtered_df.iterrows():
        if compute_distance(r[rhs], rhs_value_to_impute, df.dtypes[rhs]) <= threshold_map[rhs]:
            rows.append(i)
    filtered_df=filtered_df.loc[rows]
    possible_candidate_rows=[]
    for i, r in filtered_df.iterrows():
        insert = True
        for c in lhs:
            if compute_distance(r[c], row_to_impute[c], filtered_df[lhs].dtypes[c]) > threshold_map[c]:
                insert = False
                break
        if insert:
            possible_candidate_rows.append(i)
    try:
        filtered_df = filtered_df.loc[possible_candidate_rows]
        return filtered_df
    except:
        return None

def only_check_similar_row_by_rhs(df, lhs, rhs, index_row_to_impute, rhs_value_to_impute, threshold_map):
    filtered_df = df[[rhs]+lhs]
    row_to_impute=filtered_df.loc[index_row_to_impute]
    filtered_df=filtered_df.dropna()
    filter_rhs_time=datetime.now()
    rows=[]
    for i, r in filtered_df.iterrows():
        if compute_distance(r[rhs], rhs_value_to_impute, df.dtypes[rhs]) <= threshold_map[rhs]:
            rows.append(i)
    filtered_df=filtered_df.loc[rows]
    filter_rhs_time = datetime.now()-filter_rhs_time

    filter_lhs_time=datetime.now()

    possible_candidate_rows=[]
    to_ret=False
    for i, r in filtered_df.iterrows():
        insert = True
        for c in lhs:
            if compute_distance(r[c], row_to_impute[c], filtered_df[lhs].dtypes[c]) > threshold_map[c]:
                insert = False
                break
        if insert:
            to_ret = True
            break
    filter_lhs_time = datetime.now()-filter_lhs_time
    return to_ret


def find_attribute_to_impute(imputation_result, row):
    return imputation_result[imputation_result["riga"] == row].values[0][1]


def find_value_to_impute(initial_tuple, attr, row):
    temp=initial_tuple[initial_tuple["riga"] == row]
    temp=temp[temp["attributo"] == attr].values[0]
    return temp[2]








def attribute_improvement(attr_to_improve, minimum_score_to_reach, df, imputation_result, candidates, qrfilepath, initial_tuple_path, thresholds):
    initial_tuple = pd.read_csv(initial_tuple_path, sep=";", header=None)
    initial_tuple.columns = ["riga", "attributo", "valore"]
    dependencies = pd.read_csv(qrfilepath, sep=";")
    cols = dependencies.columns.tolist()
    col_as_attributes = {}
    cols_positions = {}
    for i, col in enumerate(cols[1:]):
        cols_positions[col] = i
        col_as_attributes[col] = Attribute(col, thresholds[i])
    dep_list = dataset_util.load_qr_file(qrfilepath)
    initial_tuple = pd.read_csv(initial_tuple_path, sep=";", header=None)
    initial_tuple.columns = ["riga", "attributo", "valore"]

    imputation_result = pd.read_csv(imputation_result, sep=";")
    imputation_result['riga'] = imputation_result['riga'].apply(lambda x: x - 1)
    imputation_result = imputation_result.sort_values(['riga'])
    mv_imputed = {}
    mv_not_imputed = {}
    for file in candidates:
        left, composed_right = file.split("_imputed_")
        right = composed_right.split("_")[0]
        row = int(right.replace(".json", ""))
        imputed = not left.endswith("not")
        with open(file, 'r') as f:
            jsonData = json.load(f)
            attr_to_impute = composed_right.replace(f"{right}_","").replace(".json", "")
            val_to_impute = find_value_to_impute(initial_tuple, attr_to_impute, row + 1)
            if imputed:
                if row not in mv_imputed:
                    mv_imputed[row]={}
                mv_imputed[row][attr_to_impute] = {"set": extract_dependency_from_json(jsonData,df.columns, "RFD"),
                                   "attribute_to_impute": attr_to_impute,
                                   "value_to_impute": val_to_impute}
            else:
                mv_not_imputed_row_set = set()
                for x in jsonData:
                    mv_not_imputed_row_set.add(extract_dependency_from_json(x,df.columns))
                if row not in mv_not_imputed:
                    mv_not_imputed[row]={}
                mv_not_imputed[row][attr_to_impute] = {"set": mv_not_imputed_row_set, "attribute_to_impute": attr_to_impute,
                                       "value_to_impute": val_to_impute}

    grouped_attrs = imputation_result.groupby("nome attributo")
    single_attribute_score = {}
    for c in df.columns:
        single_attribute_score[c]={}
        single_attribute_score[c]["used_dependencies"] = set()
        single_attribute_score[c]["score"] = 0
        single_attribute_score[c]["goodness"] = 0

    for k, v in grouped_attrs:
        score = compute_score(v)
        single_attribute_score[k] = {"score": score}
        single_attribute_score[k]["goodness"] = compute_score(v, False)

        dep = get_used_dependencies(v, mv_imputed, True)
        single_attribute_score[k]["used_dependencies"] = set()
        for d in dep:
            if d[2]==k:
                single_attribute_score[k]["used_dependencies"].add((d[0],d[1]))
            else:
                pass


    new_operation=None
    if len(single_attribute_score[attr_to_improve]["used_dependencies"])<=0:
        if dep_list.superset_on_rhs_by_name([attr_to_improve]).count_dependencies()>0:
            new_operation = ["+"] * len(df.columns)
            new_operation[cols_positions[attr_to_improve]] = ""
        else:
            new_operation = ["r"] * len(df.columns)
            new_operation[cols_positions[attr_to_improve]] = "+"
    else:
        if single_attribute_score[attr_to_improve]["score"]<minimum_score_to_reach:
            new_operation = ["+"] * len(df.columns)
            new_operation[cols_positions[attr_to_improve]] = ""

    return new_operation, compute_score(imputation_result), single_attribute_score, compute_score(imputation_result, False)


def final_score(df, imputation_result, candidates, qrfilepath, initial_tuple_path):
    initial_tuple = pd.read_csv(initial_tuple_path, sep=";", header=None)
    initial_tuple.columns = ["riga", "attributo", "valore"]
    dependencies = pd.read_csv(qrfilepath, sep=";")
    cols = dependencies.columns.tolist()

    dep_list = dataset_util.load_qr_file(qrfilepath)


    initial_tuple = pd.read_csv(initial_tuple_path, sep=";", header=None)
    initial_tuple.columns = ["riga", "attributo", "valore"]

    imputation_result = pd.read_csv(imputation_result, sep=";")
    imputation_result['riga'] = imputation_result['riga'].apply(lambda x: x - 1)
    imputation_result = imputation_result.sort_values(['riga'])
    mv_imputed = {}
    mv_not_imputed = {}
    for file in candidates:
        left, composed_right = file.split("_imputed_")
        right = composed_right.split("_")[0]
        row = int(right.replace(".json", ""))
        imputed = not left.endswith("not")
        with open(file, 'r') as f:
            jsonData = json.load(f)
            attr_to_impute = composed_right.replace(f"{right}_","").replace(".json", "")
            val_to_impute = find_value_to_impute(initial_tuple, attr_to_impute, row + 1)
            if imputed:
                if row not in mv_imputed:
                    mv_imputed[row] = {}
                mv_imputed[row][attr_to_impute] = {"set": extract_dependency_from_json(jsonData,df.columns, "RFD"),
                                   "attribute_to_impute": attr_to_impute,
                                   "value_to_impute": val_to_impute}
            else:
                mv_not_imputed_row_set = set()
                for x in jsonData:
                    mv_not_imputed_row_set.add(extract_dependency_from_json(x,df.columns))
                if row not in mv_not_imputed:
                    mv_not_imputed[row] = {}
                mv_not_imputed[row][attr_to_impute] = {"set": mv_not_imputed_row_set, "attribute_to_impute": attr_to_impute,
                                       "value_to_impute": val_to_impute}

    for k, s in mv_not_imputed.items():
        for k2, v in s.items():
            if len(v["set"]) > 1:
                continue
            possible_deps = Dependencies()
            for lhs,rhs in dep_list.items():
                rhs_names=list(map(lambda x : x.name, rhs))
                if v["attribute_to_impute"] in rhs_names:
                    rhs_to_check=None
                    for check in rhs:
                        if check.name==v["attribute_to_impute"]:
                            rhs_to_check=AttributeSet(check)
                            break
                    possible_deps[lhs]=rhs_to_check

            if len(possible_deps) <= 0:
                imputation_result.loc[(imputation_result["riga"] == k) & (imputation_result["nome attributo"] == v["attribute_to_impute"]), "valore imputato"] = "!"
                continue
            for lhs, rhs in possible_deps.items():
                rhs = list(filter(lambda x: x.name == v["attribute_to_impute"], rhs))[0]
                thrs_map = {rhs.name: rhs.threshold}
                for a in lhs:
                    thrs_map[a.name] = a.threshold
                lhs = list(map(lambda x: x.name, lhs))
                rhs = rhs.name
                possible_row = only_check_similar_row_by_rhs(df, lhs, rhs, k, v["value_to_impute"], thrs_map)
                if possible_row:
                    imputation_result.loc[(imputation_result["riga"] == k) & (imputation_result["nome attributo"] == v["attribute_to_impute"]), "valore imputato"] = "!"
                    break

    grouped_attrs = imputation_result.groupby("nome attributo")
    single_attribute_score = {}
    for k, v in grouped_attrs:
        score = compute_score(v)
        single_attribute_score[k] = {"score": score}
        single_attribute_score[k]["goodness"]=compute_score(v, False)
        dep = get_used_dependencies(v, mv_imputed, True)
        single_attribute_score[k]["used_dependencies"] = set()
        for d in dep:
            if d[2]==k:
                single_attribute_score[k]["used_dependencies"].add((d[0],d[1]))
            else:
                pass


        single_attribute_score[k]["used_dependencies"]=list(map(lambda d: f"{d[0]} -> {d[1]}".replace("[","").replace("]",""),single_attribute_score[k]["used_dependencies"]))

    return compute_score(imputation_result), single_attribute_score, imputation_result


