#!/bin/bash

python train_model_ecog_reduced.py \
-m lfads_ecog \
-d /home/ws5/manolan/data/datasets/ecog/goose_wireless/gw_250 \
-p /home/ws5/manolan/ecog_pred/hierarchical_lfads/hyperparameters/ecog/lfads_ecog_reduced_nofilt.yaml \
-o /home/ws5/manolan/data/models/pyt/lfads \
--device_num 1 \
--data_suffix ecog \
--max_epochs 3000 \
--batch_size 500 \
--seq_len 50 \
--ch_idx 10 \
-t
