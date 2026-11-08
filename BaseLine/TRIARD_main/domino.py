import os.path

import util

domino_params = {
    "folder": "domino"
}
dataset_folder = "datasets"


def run(original_file_path, separator, has_header, nullvalue="?", max_thr=0, domino_time_limit=None):
    current_cwd = os.getcwd()
    util.check_and_create_directory("domino_results")
    result_file_name_to_check=f"output_false_{str(int(max_thr))}_{os.path.basename(original_file_path)}"


    os.chdir(domino_params['folder'])

    dataset_log_folder=util.join_path(dataset_folder, "log")
    dataset_maps_folder=util.join_path(dataset_folder, "maps")
    dataset_matrices_folder=util.join_path(dataset_folder, "matrices")
    dataset_output_folder=util.join_path(dataset_folder, "SCORE")
    dataset_outputQR_folder=util.join_path(dataset_folder, "outputQR")


    dataset_matrices_complete_folder=util.join_path(dataset_matrices_folder, "complete")
    dataset_matrices_approx_folder=util.join_path(dataset_matrices_folder, "approx")
    dataset_matrices_max_thr_folder=util.join_path(dataset_matrices_folder, f"{str(max_thr)}")

    if os.path.exists(dataset_folder):
        util.drop_directory(dataset_folder)
    folder_to_check=[
        dataset_folder,
        dataset_log_folder,
        dataset_maps_folder,
        dataset_matrices_folder,
        dataset_output_folder,
        dataset_outputQR_folder,
        dataset_matrices_complete_folder,
        dataset_matrices_approx_folder,
        dataset_matrices_max_thr_folder
    ]



    for f in folder_to_check:
        util.check_and_create_directory(f)



    util.copy_file_to_folder(original_file_path, dataset_folder)

    create_matrix_cmd=f'java -Xmx{util.get_max_ram()}G -jar CreateMatrix.jar {os.path.basename(original_file_path)} {has_header} \"{separator}\" \"{nullvalue}\" false {max_thr} true'




    if domino_time_limit is not None:
        create_matrix_cmd=f"timeout {domino_time_limit} {create_matrix_cmd}"

    print(create_matrix_cmd)
    time_matrix, res=util.profile_f(os.system,create_matrix_cmd)





    domino_cmd=f'java -Xmx{util.get_max_ram()}G -jar Domino.jar {dataset_folder}{os.path.sep}{os.path.basename(original_file_path)} {str(has_header).lower()} \"{separator}\" {nullvalue} false {max_thr} true'
    if domino_time_limit is not None:
        time_matrix = int(time_matrix.total_seconds())
        rem= domino_time_limit-time_matrix
        domino_cmd=f"timeout {0 if rem<0 else rem} {domino_cmd}"


    print(domino_cmd)
    os.system(domino_cmd)
    result_file=None
    try:
        result_file=os.path.abspath(util.get_most_recent_file(dataset_outputQR_folder))
    except:
        pass

    os.chdir(current_cwd)
    if result_file is not None:
        util.copy_file_to_folder(result_file, "domino_results")
        return os.path.abspath(util.get_most_recent_file("domino_results"))
    return None


def clean():
    current_cwd = os.getcwd()

    os.chdir(domino_params['folder'])
    util.drop_directory(dataset_folder)
    os.chdir(current_cwd)
    util.drop_directory("domino_results")

