# gw_250_assess_models.py

import analyze_LFADS_ecog_performance
import time

##### code to loop over a list of directories and run the analysis script on each
# Note: analyze_LFADS_ecog_performance.analyze() takes about 2 minutes to process each model.

def get_gw_250_42ch_model_list():
    None

if __name__ == "__main__":
    t_start = time.time()
    model_dir_list = get_gw_250_42ch_model_list()
    dataset_path = r"D:\Users\mickey\Data\datasets\ecog\goose_wireless\gw_250"
    ar_model_path = r"D:\Users\mickey\Data\analysis\ar_valid_set\ar_model_dict.pkl"
    for model_dir in model_dir_list:
        hyperparameter_path = os.path.join(model_dir,'hyperparameters.yaml')
        analyze(model_dir_path, dataset_path, ar_model_path, hyperparameter_path)

    t_end = time.time()
    print(f'script complete. runtime: {t_end-t_start:0.3f}')