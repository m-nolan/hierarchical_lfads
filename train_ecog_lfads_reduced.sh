#!/bin/bash

python train_model_ecog_reduced.py \
-m lfads_ecog \
-d /home/ws5/manolan/data/datasets/ecog/goose_wireless/gw_250 \
-p /home/ws5/manolan/ecog_pred/hierarchical_lfads/hyperparameters/ecog/lfads_ecog_reduced_nofilt.yaml \
-o /home/ws5/manolan/data/models/pyt/lfads \
--data_suffix ecog \
--device_num 1 \
--max_epochs 3500 \
--batch_size 1500 \
--seq_len 50 \
--ch_idx 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 25 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 \
-t
