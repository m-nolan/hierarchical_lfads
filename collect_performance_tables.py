# collect_performance_tables.py
# 
# Aggregate a list of pandas dataframe rows from each assessed model
# "assessed model" meaning there's a file in there named "performance_table.csv"
#
# Michael Nolan
# 2021-05-02

import os
import time
import glob
import pandas as pd
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('--model_dir',type=str,default=r"D:\Users\mickey\Data\models\pyt\lfads\gw_250\lfads_ecog")

args = parser.parse_args()


def get_performance_table_file_list(model_root_path):
    # model_root_path = r"D:\Users\mickey\Data\models\pyt\lfads\gw_250\lfads_ecog"
    performance_table_file_list = glob.glob(os.path.join(model_root_path,'*','performance_table.csv'))
    return performance_table_file_list

def get_model_parameters_from_perf_table_path(perf_table_file):
    file_parts = perf_table_file.split(os.path.sep)
    model_dir_str = file_parts[-2]
    return model_dir_str

def read_and_concat_csv(table_file_list):
    table_list = []
    for file_idx, table_file in enumerate(table_file_list):
        model_dir_str = get_model_parameters_from_perf_table_path(table_file)
        table_row = pd.read_csv(table_file)
        table_row['model_dir_name'] = model_dir_str
        table_list.append(table_row)
    table_cat = pd.concat(table_list)
    return table_cat

def collect_performance_tables(model_root_path):
    perf_table_file_list = get_performance_table_file_list(model_root_path)
    print(f'Gathering performance records from models in:\t{model_root_path}')
    print(f'{len(perf_table_file_list)} model performance files found:\n{perf_table_file_list}')
    perf_table = read_and_concat_csv(perf_table_file_list)
    cat_table_file_path = os.path.join(model_root_path,'performance_table_all.csv')
    perf_table.to_csv(cat_table_file_path)

if __name__ == "__main__":
    t_start = time.time()
    collect_performance_tables(args.model_dir)
    t_end = time.time()
    print(f'Analysis complete. Runtime:\t{t_end-t_start}')
