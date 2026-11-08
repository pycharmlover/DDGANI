import dependencies_util
import pandas as pd
import os
import util
import shutil
import json
from dependencies_util import Attribute
import numpy as np
import random
inject_params = {
    "generate_dataset_folder": "generate_dataset",
    "jar_name": "missing_data_generator.jar",
    "percentage": 2,
    "version": 1
}

discovery_params = {
    "discovery_folder": "discovery",
    "jar_name": "dime.jar",
    "resfolder": "resfolder",
    "qroutfolder": "qroutfolder"
}

def load_qr_file(qr_file):
    dependencies_file = pd.read_csv(qr_file, sep=";")
    cols = dependencies_file.columns.tolist()
    dependencies = dependencies_util.Dependencies()
    for r in dependencies_file.iterrows():
        rhs_label = r[1]["RHS"]
        rhs = []
        lhs = []
        for i in range(1, len(r[1])):
            if r[1].iloc[i] != "?":
                attr = Attribute(cols[i], float(r[1].iloc[i]))
                if cols[i] == rhs_label:
                    rhs.append(attr)
                else:
                    lhs.append(attr)
        dependencies[dependencies_util.AttributeSet(lhs)] = dependencies_util.AttributeSet(rhs)
    return dependencies

def load_datasets_portions(df: pd.DataFrame):
    df_without_null = df.dropna()
    df_with_null = df[df.isna().any(axis=1)]
    return df_without_null, df_with_null


def inject_coordinates(dataset_file_path, dataset_separator, dataset_has_header, dataset_null_char, coordinates):
    current_cwd = os.getcwd()

    dataset_filename = os.path.basename(dataset_file_path)
    dataset_name = dataset_filename.split('.')[0]
    dataset_name=f"{dataset_name}_{str(len(coordinates))}_1.csv"
    initial_tuples_folder = util.join_path(inject_params['generate_dataset_folder'], "InitialTuples")
    missing_dataset_folder = util.join_path(inject_params['generate_dataset_folder'], "MissingDataset")
    full_dataset_folder = util.join_path(inject_params['generate_dataset_folder'], "FullDataset")

    util.check_and_create_directory(inject_params['generate_dataset_folder'])
    util.check_and_create_directory(initial_tuples_folder)
    util.check_and_create_directory(missing_dataset_folder)
    util.check_and_create_directory(full_dataset_folder)


    shutil.copyfile(dataset_file_path, util.join_path(full_dataset_folder, dataset_filename))

    os.chdir(inject_params['generate_dataset_folder'])
    print(os.getcwd())
    df=pd.read_csv(util.join_path("FullDataset", dataset_filename), sep=dataset_separator)
    initial_tuples_list=[]
    for attr in coordinates:
        r=random.choice(list(range(len(df))))
        value=df[attr].iloc[r]
        initial_tuples_list.append([r+1, attr, value])
        df.loc[r,attr]="?"
    initial_tuples_df=pd.DataFrame(initial_tuples_list)
    df.to_csv(util.join_path("MissingDataset", dataset_name), sep=dataset_separator, index=False)
    initial_tuples_df.to_csv(util.join_path("InitialTuples", dataset_name), sep=dataset_separator, index=False, header=False)


    os.chdir(current_cwd)

    return {"df":pd.read_csv(
        util.get_most_recent_file(initial_tuples_folder),
        sep=dataset_separator,
        header=0 if dataset_has_header else None,
        na_values=dataset_null_char
    ), "path":util.get_most_recent_file(initial_tuples_folder)}, {"df":pd.read_csv(
        util.get_most_recent_file(missing_dataset_folder),
        sep=dataset_separator,
        header=0 if dataset_has_header else None,
        na_values=dataset_null_char
    ), "path":util.get_most_recent_file(missing_dataset_folder)}





def inject_mv(dataset_file_path, dataset_separator, dataset_has_header, dataset_null_char, percentage=None):
    current_cwd = os.getcwd()

    dataset_filename = os.path.basename(dataset_file_path)
    dataset_name = dataset_filename.split('.')[0]

    initial_tuples_folder = util.join_path(inject_params['generate_dataset_folder'], "InitialTuples")
    missing_dataset_folder = util.join_path(inject_params['generate_dataset_folder'], "MissingDataset")
    full_dataset_folder = util.join_path(inject_params['generate_dataset_folder'], "FullDataset")

    util.check_and_create_directory(inject_params['generate_dataset_folder'])
    util.check_and_create_directory(initial_tuples_folder)
    util.check_and_create_directory(missing_dataset_folder)
    util.check_and_create_directory(full_dataset_folder)


    shutil.copyfile(dataset_file_path, util.join_path(full_dataset_folder, dataset_filename))

    os.chdir(inject_params['generate_dataset_folder'])
    print(os.getcwd())
    run_string = f'java -jar {inject_params["jar_name"]} "{dataset_separator}" {dataset_name} {inject_params["percentage"] if percentage is None else percentage} {dataset_has_header} {inject_params["version"]}'
    print(run_string)
    os.system(run_string)

    os.chdir(current_cwd)

    return {"df":pd.read_csv(
        util.get_most_recent_file(initial_tuples_folder),
        sep=dataset_separator,
        header=0 if dataset_has_header else None,
        na_values=dataset_null_char
    ), "path":util.get_most_recent_file(initial_tuples_folder)}, {"df":pd.read_csv(
        util.get_most_recent_file(missing_dataset_folder),
        sep=dataset_separator,
        header=0 if dataset_has_header else None,
        na_values=dataset_null_char
    ), "path":util.get_most_recent_file(missing_dataset_folder)}

def save_to_qr(dependencies, header, qroutfolder, output_file):
    result_df = pd.DataFrame(index=np.arange(dependencies.count_dependencies()), columns=["RHS"] + header)
    result_df[:] = "?"
    iterator = 0
    list_dep = list(map(lambda x: {"lhs": x[0], "rhs": x[1]}, dependencies.items()))
    for i in range(len(list_dep)):
        dep = list_dep[i]
        for rhs in dep["rhs"]:
            result_df.loc[iterator, "RHS"] = rhs.name
            result_df.loc[iterator, rhs.name] = rhs.threshold
            for lhs in dep["lhs"]:
                result_df.loc[iterator, lhs.name] = lhs.threshold
            iterator += 1
    util.check_and_create_directory(qroutfolder)
    result_df.to_csv(util.join_path(qroutfolder, output_file), sep=';', index=False)
    return os.path.join(qroutfolder, output_file)


def export_to_qr(filepath, header, qroutfolder):
    dependencies_loader= dependencies_util.DependenciesLoader()
    with open(filepath) as f:
        deps = json.load(f)
        for d in deps:
            dependencies_loader.add_dependency(d)
    output_file = (os.path.basename(filepath).split(".")[0]) + ".csv"
    return save_to_qr(dependencies_loader.dependencies, header, qroutfolder, output_file)

def discovery(dataset_file_path, dataset_separator, dataset_has_header, dataset_null_char, thrs, columns):
    current_cwd = os.getcwd()
    util.check_and_create_directory(discovery_params['discovery_folder'])
    util.check_and_create_directory(discovery_params['qroutfolder'])
    os.chdir(discovery_params['discovery_folder'])
    util.check_and_create_directory(discovery_params['resfolder'])
    util.clean_directory(discovery_params['resfolder'])
    util.clean_directory("log")
    run_string = (f"java -Xmx{util.get_max_ram()}G -jar {discovery_params['jar_name']} {dataset_file_path} {' '.join(map(lambda x: str(x),thrs))} -s \"{dataset_separator}\" -r {discovery_params['resfolder']} "
                  f"-head {dataset_has_header} -pd true -n {dataset_null_char} > {util.get_null_redirect_output()} 2>&1")
    print(run_string)
    os.system(run_string)
    result_file = util.get_file_first_file(discovery_params['resfolder'])
    os.chdir(current_cwd)

    toret = export_to_qr(util.join_path(discovery_params['discovery_folder'],result_file), columns,
                         discovery_params['qroutfolder'])
    util.clean_directory(util.join_path(discovery_params['discovery_folder'],discovery_params['resfolder']))
    util.clean_directory(util.join_path(discovery_params['discovery_folder'],"log"))
    return toret

def get_dtypes(df):
    dtypes = ["D"] * len(df.columns)
    col_iter=0

    for c in df.columns:
        if len(df[c].unique()) == 2:
            dtypes[col_iter] = "B"
        elif df[c].dtype == "object":
            if df[c].str.len().max() == 1:
                dtypes[col_iter] = "C"
        else:
            pass
        col_iter += 1
    return dtypes

def clean():
    current_cwd = os.getcwd()
    os.chdir(inject_params['generate_dataset_folder'])
    util.drop_directory("FullDataset")
    util.drop_directory("InitialTuples")
    util.drop_directory("MissingDataset")
    os.chdir(current_cwd)
    os.chdir(discovery_params['discovery_folder'])
    util.drop_directory("log")
    util.drop_directory(discovery_params["resfolder"])
    os.chdir(current_cwd)
    util.drop_directory(discovery_params["qroutfolder"])



