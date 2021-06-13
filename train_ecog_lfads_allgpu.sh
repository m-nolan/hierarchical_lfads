#!/bin/bash

python train_model_ecog_reduced.py \
-m lfads_ecog \
-d /home/ws5/manolan/data/datasets/ecog/goose_wireless/gw_250 \
-p /home/ws5/manolan/ecog_pred/hierarchical_lfads/hyperparameters/ecog/lfads_ecog_encdecsweep.yaml \
-o /home/ws5/manolan/data/models/pyt/lfads \
--data_suffix ecog_fl80u100 \
--max_epochs 2000 \
--batch_size 1000 \
--seq_len 50 \
--loss mse \
--device_num 1 \
-t
