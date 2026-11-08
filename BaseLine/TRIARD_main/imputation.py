import util
import os
import shutil
import pandas as pd
imputation_params = {
    "imputation_folder": "imputation",
    "jar_name": "renuver.jar"
}

folders = ["Candidates",
           "Dataset",
           "ImputationResults",
           "InitialTuples",
           "KeyRFDs",
           "Logs",
           "NonKeyRFDs",
           "RFD"]



def impute(dataset_mv, initial_tuples, qrfilepath, dtypes, dataset_separator, time_limit=None):

    current_cwd=os.getcwd()
    os.chdir(imputation_params['imputation_folder'])
    for folder in folders:
        util.check_and_create_directory(folder)
        util.clean_directory(folder)
    os.chdir(current_cwd)
    dataset_mv_filename = os.path.basename(dataset_mv)
    initial_tuples_filename = os.path.basename(initial_tuples)
    qr_filename = os.path.basename(qrfilepath)
    dataset_mv_new_path=os.path.join(imputation_params['imputation_folder'],"Dataset",dataset_mv_filename)
    initial_tuples_new_path=os.path.join(imputation_params['imputation_folder'],"InitialTuples",initial_tuples_filename)
    qr_new_path=os.path.join(imputation_params['imputation_folder'],"RFD",qr_filename)
    shutil.copyfile(dataset_mv, dataset_mv_new_path)
    shutil.copyfile(initial_tuples, initial_tuples_new_path)
    shutil.copyfile(qrfilepath, qr_new_path)
    qr_df=pd.read_csv(qr_new_path, sep=dataset_separator)
    all_thrs=set()
    for c in qr_df.columns[1:]:
        all_thrs.update(list(qr_df[c].unique()))
    if '?' in all_thrs:
        all_thrs.remove('?')
    all_thrs=list(map(lambda x:round(float(x),10), all_thrs))
    if len(all_thrs)>0:
        max_thr=max(all_thrs)
    else:
        max_thr=0
    os.chdir(imputation_params['imputation_folder'])

    run_string = (f"java -Xmx{util.get_max_ram()}G -jar {imputation_params['jar_name']} {','.join(dtypes)} "
                  f"{dataset_mv_filename.split('.')[0]} \"{dataset_separator}\" "
                  f"{qr_filename} {max_thr} > {util.get_null_redirect_output()}") #
    if time_limit is not None:
        run_string=f"timeout {time_limit} {run_string}"
    print(run_string)
    os.system(run_string)
    candidates = (util.get_all_files("Candidates"))
    candidates = list(map(lambda f: util.join_path(imputation_params['imputation_folder'], f), candidates))
    result = util.get_file_first_file("ImputationResults")
    if result:
        result = util.join_path(imputation_params['imputation_folder'], result)
    os.chdir(current_cwd)
    return result,candidates

def clean():
    current_cwd = os.getcwd()
    os.chdir(imputation_params['imputation_folder'])
    for folder in folders:
        util.drop_directory(folder)
    os.chdir(current_cwd)