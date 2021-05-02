# gw_250_assess_models.py

import analyze_LFADS_ecog_performance
import os
from glob import glob
import time

##### code to loop over a list of directories and run the analysis script on each
# Note: analyze_LFADS_ecog_performance.analyze() takes about 2 minutes to process each model.

dry_run = False
overwrite = False

def get_gw_250_42ch_model_list():
    model_parent_directory = r"D:\Users\mickey\Data\models\pyt\lfads\gw_250\lfads_ecog"
    model_re = "cenc[0-9]*_cont[0-9]*_fact[0-9]*_genc1024_gene1024_glat1024_nch42_seqlen[0-9]*_ulat[0-9]*_orion*"
    model_dir_list = glob(os.path.join(model_parent_directory,model_re))
    return model_dir_list

if __name__ == "__main__":
    t_start = time.time()
    model_dir_list = get_gw_250_42ch_model_list()
    n_model = len(model_dir_list)
    dataset_path = r"D:\Users\mickey\Data\datasets\ecog\goose_wireless\gw_250"
    ar_model_path = r"D:\Users\mickey\Data\analysis\ar_valid_set\ar_model_dict.pkl"
    for idx, model_dir_path in enumerate(model_dir_list):
        t_model_start = time.time()
        hyperparameter_path = os.path.join(model_dir_path,'hyperparameters.yaml')
        print(f'Assessing model {idx+1}/{n_model} at {model_dir_path}:\n')
        if dry_run:
            print('dry run, skipping analysis.')
        else:
            # check to see if the performance table already exists
            perf_table_exists = os.path.exists(os.path.join(model_dir_path,
                'performance_table.csv'))
            run_analysis = (not perf_table_exists) or overwrite
            if run_analysis:
                try:
                    analyze_LFADS_ecog_performance.analyze(model_dir_path, dataset_path, 
                        ar_model_path, hyperparameter_path)
                except:
                    # this SHOULD be handled by logging, but I don't know how to do that.
                    print('analysis error! Skipping for now, fix and re-run.')
            else:
                print(f'Analysis skipped: Performance Table Exists: {perf_table_exists}\tOverwrite Mode: {overwrite}')
        t_model_end = time.time()
        print(f'Model assessment time:\t{t_model_end-t_model_start:0.3f}')
    t_end = time.time()
    print(f'script complete. runtime:\t{t_end-t_start:0.3f}')