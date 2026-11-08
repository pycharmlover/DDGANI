import argparse
import statistics

import pandas as pd
from dependencies_util import DependenciesLoader, DependenciesFormatter, Dependencies, AttributeSet, Attribute
import os


import analyze_imputation
import dataset_util
import imputation
import load_metadata
import util
import domino
from datetime import datetime
import time
import json

def get_new_threshold(operation, current_value, distances):
    if operation=="": return current_value
    el_index=distances.index(current_value)
    if operation=="+":
        if el_index<len(distances)-1:
            return distances[el_index+1]
        else:
            return distances[len(distances)-1]

    if operation=="-":
        if el_index>0:
            return distances[el_index-1]
        else:
            return distances[0]
    if operation=="r":
        return distances[0]
    print("Threshold error!",(operation, current_value, distances))
    return None


def adjust_threshold(cols_order, current, mods, stats):
    to_ret=[]
    i=0
    for c in cols_order:
        to_ret.append(get_new_threshold(mods[i],current[i],stats[c]))
        i+=1
    return to_ret


def reduce_by_dominance(dependencies_set):
    #return dependencies_set
    return dependencies_set.apply_dominance()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='TRIAD',
        description='Run imputation using data dependencies and adaptive threshold tuning',
        epilog='Developed for automated data imputation research.')

    parser.add_argument('dataset_file_path', help="Path to the dataset file (CSV with semicolon separator) or folder")

    parser.add_argument('--remove_log', action='store_true',
                        help='Prevent saving intermediate logs and results during processing')

    parser.add_argument('--dataset_has_not_header',   action='store_true',
                        help='Specify if the dataset has not a header row')

    parser.add_argument('--dataset_null_char', type=str, default='?',
                        help='Character used to represent missing values in the dataset')

    parser.add_argument('--output_folder', type=str, default='output',
                        help='Folder where output files and logs will be stored')

    parser.add_argument('--increasing_steps', type=int, nargs='*', default=[],
                        help='List of increasing steps for each attribute')

    parser.add_argument('--max_similarity_values', type=float, nargs='*', default=[],
                        help='List of max similarity values for each attribute')

    parser.add_argument('--iterations', type=int, default=7,
                        help='Maximum number of tuning iterations per attribute')

    parser.add_argument('--min_score_to_reach', type=float, default=0.5,
                        help='Minimum score threshold to stop the optimization for an attribute')

    parser.add_argument('--prevent_use_previous_results_at_each_step',   type=bool, default=True,
                        help='Prevent the use dependencies from previous steps to improve the next iterations')

    parser.add_argument('--missing_percentage_generator', type=float, default=1,
                        help='Minimum score threshold to stop the optimization for an attribute')

    parser.add_argument('--prevent_run_domino', action='store_true',
                        help='Prevent run of imputation with domino')

    parser.add_argument('--time_limit', type=int, default=None,
                        help='Time limit (seconds)')

    args = parser.parse_args()

    file_path = args.dataset_file_path
    save_log = not args.remove_log
    dataset_has_header = not args.dataset_has_not_header
    dataset_null_char = args.dataset_null_char
    output_folder = args.output_folder
    iterations = args.iterations
    min_score_to_reach = args.min_score_to_reach
    use_previous_results_at_each_step = not args.prevent_use_previous_results_at_each_step
    time_limit=args.time_limit

    missing_percentage_generator = args.missing_percentage_generator
    run_domino = not args.prevent_run_domino


    files_to_run=[]
    if file_path.endswith(".csv"):
        files_to_run=[file_path]
    elif util.is_dir(file_path):
        files_to_run=util.get_all_files(file_path, filter="*.csv")
    else:
        print("Dataset file is not a CSV file or directory")
        exit(0)

    for dataset_file_path in files_to_run:
        increasing_steps = args.increasing_steps
        max_similarity_values = args.max_similarity_values
        preprocessing_start_time = datetime.now()
        with open(dataset_file_path) as f:
            first_line = f.readline()
            comma_len = len(first_line.split(","))
            semicolon_len = len(first_line.split(";"))
            dataset_separator = "," if comma_len > semicolon_len else ";"

        if dataset_separator!=";":
            print("The semicolon is required as CSV separator")
            continue



        dataset_filename = os.path.basename(dataset_file_path)
        dataset_name = dataset_filename.split('.')[0]

        util.check_and_create_directory(output_folder)
        current_exp_folder=util.join_path(output_folder,f"{dataset_filename}_{int(round(time.time() * 1000))}")
        util.check_and_create_directory(current_exp_folder)

        if save_log:
            current_imputation_result_folder=util.join_path(current_exp_folder,f"imputation_results")
            current_dependencies_result_folder=util.join_path(current_exp_folder,f"dependencies")
            util.check_and_create_directory(current_imputation_result_folder)
            util.check_and_create_directory(current_dependencies_result_folder)


        print(f"Read dataset file '{dataset_filename}'")
        read_time, df = util.profile_f(pd.read_csv,
            dataset_file_path,
            sep=dataset_separator,
            header=0 if dataset_has_header else None,
            na_values=dataset_null_char
        )
        print(f"Read time: {read_time}")

        dataset_has_null_values=df.isna().any().any()

        if not increasing_steps:
            increasing_steps = [-1] * len(df.columns)
        if not max_similarity_values:
            max_similarity_values = [-1] * len(df.columns)


        if dataset_has_null_values:
            train_set, test_set = dataset_util.load_datasets_portions(df)
            print(f"Dataset already contains null values: {len(test_set)}")
            continue
        else:
            print(f"Dataset not contains null values: injecting")
            df_initial_tuple, df_injected_dataset = dataset_util.inject_mv(dataset_file_path, dataset_separator, dataset_has_header,
                                                                     dataset_null_char, missing_percentage_generator)
            train_set, test_set = dataset_util.load_datasets_portions(df_injected_dataset["df"])





        print(f"Dataset rows with null values: {len(test_set)}")

        print(f"Extracting metadata")
        dtypes=dataset_util.get_dtypes(train_set)
        #distance_map, stats_map=load_metadata.load_metadata(df)
        stats_map=load_metadata.load_metadata_reduced(train_set)
        ordered_distances={}
        medians=[]
        domino_max_thr=0
        for k,v in stats_map.items():
            median=int(statistics.median(v))
            medians.append(median)
            if median>domino_max_thr:
                domino_max_thr=median
        domino_max_thr=int(max(medians))

        for stats_map_i, (stats_map_k, stats_map_v) in enumerate(stats_map.items()):
            increasing_step = increasing_steps[stats_map_i]
            max_similarity = max_similarity_values[stats_map_i]
            filtered=stats_map_v.copy()
            if max_similarity > -1:
                filtered = [x for x in stats_map_v if x <= max_similarity]
            if increasing_step > 1:
                filtered = [x for j, x in enumerate(filtered) if j % increasing_step == 0]
            ordered_distances[stats_map_k] = filtered


        preprocessing_end_time = datetime.now()

        with open(util.join_path(current_exp_folder, "preprocessing.json"), "w") as outfile:
            outfile.write(json.dumps({
                "preprocessing_tot_time":str(preprocessing_end_time-preprocessing_start_time),
                "preprocessing_start_time":str(preprocessing_start_time),
                "preprocessing_end_time":str(preprocessing_end_time),
                "iterations": iterations,
                "min_score_to_reach": min_score_to_reach,
                "use_previous_results_at_each_step": use_previous_results_at_each_step,
                "missing_percentage_generator":missing_percentage_generator,
                "increasing_steps":increasing_steps,
                "max_similarity_values": max_similarity_values
            }, indent=4))

        train_set_filename=f"{dataset_name}_train_set.csv"
        train_set_path=util.join_path(current_exp_folder, train_set_filename)
        print(f"Save train set in: {train_set_path}")
        train_set.to_csv(train_set_path, sep=dataset_separator,index=False, header=True)


        print(f"Inject null values for train \"thresholds\": {train_set_path}")
        train_set_injected_initial_tuple, train_set_injected_dataset = dataset_util.inject_mv(train_set_path, dataset_separator, dataset_has_header, dataset_null_char, missing_percentage_generator)


        df_initial_tuple["df"].to_csv(util.join_path(current_exp_folder, "df_initial_tuple.csv"), index=None, sep=";")
        df_injected_dataset["df"].to_csv(util.join_path(current_exp_folder, "df_injected_dataset.csv"), index=None, sep=";")
        train_set_injected_initial_tuple["df"].to_csv(util.join_path(current_exp_folder, "train_set_injected_initial_tuple.csv"), index=None, sep=";")
        train_set_injected_dataset["df"].to_csv(util.join_path(current_exp_folder, "train_set_injected_dataset.csv"), index=None, sep=";")

        discovery_set, null_discovery_set = dataset_util.load_datasets_portions(train_set_injected_dataset["df"])
        discovery_set_filename = f"{dataset_name}_discovery_set.csv"
        discovery_set_path = util.join_path(current_exp_folder, discovery_set_filename)
        discovery_set.to_csv(discovery_set_path, sep=dataset_separator,index=False, header=True)


        if run_domino:
            domino_start_time = datetime.now()

            print("Run DOMINO...")
            domino_qr_path=domino.run(os.path.abspath(train_set_path), dataset_separator, dataset_has_header, dataset_null_char, max_thr=domino_max_thr, domino_time_limit=time_limit)
            if domino_qr_path is not None:
                print("Impute through DOMINO RFDCs...")

                remain_domino_time_for_imputation = None
                if time_limit:
                    remain_domino_time_for_imputation=int(time_limit-(datetime.now() - domino_start_time).total_seconds())
                domino_imputation_time, (domino_imputation_result, domino_candidates) = util.profile_f(imputation.impute,df_injected_dataset["path"], df_initial_tuple["path"],
                                                                      domino_qr_path,
                                                                      dtypes,
                                                                      dataset_separator, time_limit= 0 if time_limit and remain_domino_time_for_imputation<0 else remain_domino_time_for_imputation)
                if domino_imputation_result:
                    domino_end_time = datetime.now()
                    print("Load DOMINO scores...")
                    domino_analyze_imputation_time,(domino_thresholds_score,
                     domino_single_attribute_score, domino_imputation_result_df) = util.profile_f(analyze_imputation
                                                                          .final_score,df_injected_dataset["df"],
                                                                                       domino_imputation_result,
                                                                                       domino_candidates, domino_qr_path,
                                                                                       df_initial_tuple["path"]
                                                                                       )



                    domino_imputation_result_df.to_csv(util.join_path(current_exp_folder,"domino_imputation_result.csv"), sep=";", index=False)
                    util.copy_file(domino_qr_path, util.join_path(current_exp_folder,"domino_qr_path.csv"))

                    domino_results=json.dumps({
                        "domino_total": str(domino_end_time - domino_start_time),
                        "domino_start_time": str(domino_start_time),
                        "domino_end_time": str(domino_end_time),
                        "domino_thresholds_score": domino_thresholds_score,
                        "domino_goodness": analyze_imputation.compute_score(domino_imputation_result_df, count_impossible=False),
                        "domino_single_attribute_score": domino_single_attribute_score,
                        "domino_imputation_time": str(domino_imputation_time),
                        "domino_analyze_imputation_time":str(domino_analyze_imputation_time)
                    }, indent=4)
                    with open(util.join_path(current_exp_folder,"domino_score.json"), "w") as outfile:
                        outfile.write(domino_results)

        triard_start_time = datetime.now()



        initial_thresholds=[0]*len(df.columns)

        list_of_iteration_results=[]
        reach_stop=[]
        best_attributes_score={

        }

        combination_tested=set()


        current_thresholds_score=0
        time_limit_reached=False
        for column_j, attr in enumerate(list(df.columns)):
            iteration_i=0

            if time_limit_reached:
                print("Time limit reached")
                break

            print(f"{'=' * 10}Start: {attr}{'=' * 10}\n\n")

            current_thresholds=initial_thresholds
            if attr in best_attributes_score and best_attributes_score[attr]["score"]>=min_score_to_reach: continue
            while True:
                if time_limit:
                    time_limit_reached = (datetime.now() - triard_start_time).total_seconds() > time_limit
                    if time_limit_reached:
                        break
                if iteration_i>=iterations:break
                attr_set = AttributeSet([Attribute(n, current_thresholds[i]) for i, n in enumerate(df.columns)])
                if attr_set in combination_tested:
                    print(f"{'^'*30} Discard combination: {attr_set} {'^'*30}")
                    thr_possible_update=[""]*len(df.columns)
                    thr_possible_update[column_j] = "+"
                    new_thrs=adjust_threshold(df.columns, current_thresholds, thr_possible_update, ordered_distances)
                    if new_thrs==current_thresholds:
                        thr_possible_update = ["+"] * len(df.columns)
                        thr_possible_update[column_j] = ""
                        new_thrs=adjust_threshold(df.columns, current_thresholds, thr_possible_update, ordered_distances)
                    if new_thrs == current_thresholds:
                        print(f"{'+'*30} Reach max {'+'*30}")
                        break
                    current_thresholds=new_thrs
                    continue
                else:
                    combination_tested.add(attr_set)
                print(f"\n{'=' * 10}Elapsed time: {str(datetime.now() - triard_start_time)}{'=' * 10}")
                print(f"{iteration_i+1}/{iterations} - {attr} ({column_j+1}/{len(df.columns)}) - Run discovery on dataset... ")
                total_discovery_time, qr_out = util.profile_f(dataset_util.discovery,os.path.abspath(discovery_set_path), dataset_separator, dataset_has_header, dataset_null_char, current_thresholds, list(df.columns))
                print(f"Discovery time: {total_discovery_time}")
                if save_log:
                    util.copy_file(qr_out, util.join_path(current_dependencies_result_folder,f"{attr}_{iteration_i}.csv"))
                if use_previous_results_at_each_step:
                    increased_qr=dataset_util.load_qr_file(qr_out)
                    for k, v in best_attributes_score.items():
                        for lhs,rhs in v["used_dependencies"].items():
                            increased_qr[lhs] = rhs
                    increased_qr=reduce_by_dominance(increased_qr)
                    #increased_qr=increased_qr.superset_on_rhs_by_name(attr)
                    qr_out = dataset_util.save_to_qr(increased_qr, list(df.columns), "qroutfolder", os.path.basename(qr_out))

                print(f"{iteration_i + 1}/{iterations} - {attr} ({column_j + 1}/{len(df.columns)}) - Run imputation...")
                triard_time_current_imputation = None
                if time_limit:
                    triard_time_current_imputation = int(
                        time_limit - (datetime.now() - triard_start_time).total_seconds())
                total_imputation_time, (imputation_result, candidates) = util.profile_f(imputation.impute,train_set_injected_dataset["path"],
                                                                  train_set_injected_initial_tuple["path"],
                                                                  qr_out,
                                                                  dtypes,
                                                                  dataset_separator, time_limit=0 if time_limit and triard_time_current_imputation<0 else triard_time_current_imputation)
                print(f"Imputation time: {total_imputation_time}")

                if save_log and imputation_result: util.copy_file(imputation_result, util.join_path(current_imputation_result_folder,f"{attr}_{iteration_i}.csv"))
                if imputation_result:
                    print(
                        f"{iteration_i + 1}/{iterations} - {attr} ({column_j + 1}/{len(df.columns)}) - Imputation results analysis...")

                    (thr_possible_update,
                     new_thresholds_score,
                     single_attribute_score, goodness) = (analyze_imputation
                                                .attribute_improvement(attr,
                                                                              min_score_to_reach,
                                                                              train_set_injected_dataset["df"],
                                                                              imputation_result,
                                                                              candidates, qr_out,
                                                                              train_set_injected_initial_tuple["path"],
                                                                              current_thresholds
                                                                              ))

                    for k,v in single_attribute_score.items():
                        if k in best_attributes_score:
                            if best_attributes_score[k]["score"]<v["score"]:
                                best_attributes_score[k]["score"]=v["score"]
                                best_attributes_score[k]["goodness"]=v["goodness"]
                                best_attributes_score[k]["used_dependencies"] = {}
                                best_attributes_score[k]["used_dependencies"].update(v["used_dependencies"])
                            elif best_attributes_score[k]["score"]==v["score"]:
                                best_attributes_score[k]["used_dependencies"].update(v["used_dependencies"])
                        else:
                            best_attributes_score[k]={}
                            best_attributes_score[k]["score"]=v["score"]
                            best_attributes_score[k]["goodness"]=v["goodness"]
                            best_attributes_score[k]["used_dependencies"]={}
                            best_attributes_score[k]["used_dependencies"].update(v["used_dependencies"])
                    if save_log: list_of_iteration_results.append(
                            (attr,iteration_i,current_thresholds,thr_possible_update,
                         new_thresholds_score,
                         single_attribute_score, total_discovery_time, total_imputation_time))
                    print(f"{'*' * 10}Iteration: {iteration_i+1} ({'{:.4f}'.format(new_thresholds_score)}) ({'{:.4f}'.format(best_attributes_score[attr]['score'])}){'*' * 10}\n")
                    if (best_attributes_score[attr]["score"]>=min_score_to_reach
                            or best_attributes_score[attr]["score"]>single_attribute_score[attr]["score"])\
                            or current_thresholds_score>new_thresholds_score:
                        print(f"best_attributes_score: {best_attributes_score[attr]['score']}, single_attribute_score: {single_attribute_score[attr]['score']}")
                        print(f"current_thresholds_score: {current_thresholds_score}, new_thresholds_score: {new_thresholds_score}")
                        print("BREAK")
                        break
                    if thr_possible_update is None:
                        break
                    current_thresholds_score=new_thresholds_score
                    current_thresholds = adjust_threshold(df.columns, current_thresholds, thr_possible_update, ordered_distances)
                iteration_i+=1
            print(f"\n{'=' * 10} Finish: {attr} {'=' * 10}\n")

        if not time_limit_reached:
            triard_time_for_final_imputation = None
            if time_limit:
                triard_time_for_final_imputation = int(
                    time_limit - (datetime.now() - triard_start_time).total_seconds())
            all_possible_dependencies=Dependencies()
            for k, v in best_attributes_score.items():
                for lhs,rhs in v["used_dependencies"].items():
                    all_possible_dependencies[lhs]=rhs

            dependencies_final=reduce_by_dominance(all_possible_dependencies)


            triard_end_time = datetime.now()


            to_save=',\n'.join(list(map(lambda d: f'\"{str(d[0])} -> {str(d[1])}\"'.replace("[","").replace("]",""), dependencies_final.items())))
            print(to_save)

            util.check_and_create_directory("final_dependencies_set")
            final_path="final_dependencies_set/final_result.json"
            with open(final_path, "w") as f:
                f.write(f'[{to_save}]')

            final_qr=dataset_util.export_to_qr(final_path, list(df.columns),'qroutfolder')


            if not dataset_has_null_values:
                final_imputation_time,(imputation_result, candidates)=util.profile_f(imputation.impute, df_injected_dataset["path"], df_initial_tuple["path"],
                                                                      final_qr,
                                                                      dtypes,
                                                                      dataset_separator, time_limit=0 if time_limit and triard_time_for_final_imputation<0 else triard_time_for_final_imputation)
                print(f"Final imputation time: {final_imputation_time}")
                triard_end_time = datetime.now()
                if imputation_result:
                    print("Check final result score....")
                    triard_analyze_imputation_time,(current_thresholds_score,
                     single_attribute_score, imputation_result_df) = util.profile_f(analyze_imputation
                                                .final_score,df_injected_dataset["df"],
                                                                       imputation_result,
                                                                       candidates, final_qr,
                                                                       df_initial_tuple["path"]
                                                                       )

                    imputation_result_df.to_csv(util.join_path(current_exp_folder, "triard_imputation_result.csv"), sep=";",
                                                       index=False)
                    triard_results = json.dumps({
                        "triard_total": str(triard_end_time - triard_start_time),
                        "triard_start_time": str(triard_start_time),
                        "triard_end_time": str(triard_end_time),
                        "triard_thresholds_score": current_thresholds_score,
                        "triard_goodness": analyze_imputation.compute_score(imputation_result_df, count_impossible=False),
                        "triard_single_attribute_score": single_attribute_score,
                        "final_imputation_time":str(final_imputation_time),
                        "triard_analyze_imputation_time":str(triard_analyze_imputation_time)
                    }, indent=4)
                    if save_log:
                        iteration_save_obj={}
                        for iteration in list_of_iteration_results:

                            obj_to_save={
                                "attr": iteration[0],
                                "current_thresholds": list(map(lambda x: int(x),iteration[2])),
                                "thr_possible_update": iteration[3],
                                "current_thresholds_score": iteration[4],
                                "discovery_time": str(iteration[6]),
                                "imputation_time": str(iteration[7])
                            }
                            for k,v in iteration[5].items():
                                casted_dep=list(map(lambda d: f"{d[0]} -> {d[1]}".replace("[", "").replace("]", ""),
                                             v["used_dependencies"]))
                                iteration[5][k]["used_dependencies"]=casted_dep
                            obj_to_save["single_attribute_score"]=iteration[5]
                            iteration_save_obj[f"{iteration[0]}_iteration_{iteration[1]}"]=obj_to_save
                        iteration_save_obj=json.dumps(iteration_save_obj, indent=4)
                        with open(util.join_path(current_exp_folder, "triard_iterations.json"), "w") as outfile:
                            outfile.write(iteration_save_obj)
                        with open(util.join_path(current_exp_folder, "triard_all_dep.json"), "w") as outfile:
                            outfile.write(to_save)





                    with open(util.join_path(current_exp_folder, "triard_score.json"), "w") as outfile:
                        outfile.write(triard_results)
                else: print("Time limit reached on final imputation")
            else:
                pass
            print(f"{'='*30}")

        dataset_util.clean()
        imputation.clean()
        util.drop_directory("final_dependencies_set")
        if run_domino:
            domino.clean()
