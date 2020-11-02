#!/bin/bash

python train_model_ecog_reduced.py \
-m lfads_ecog \
-d /home/ws5/manolan/data/datasets/ecog/goose_wireless/gw_250_fl0u10 \
-p /home/ws5/manolan/ecog_pred/hierarchical_lfads/hyperparameters/ecog/lfads_ecog_reduced.yaml \
-o /home/ws5/manolan/data/models/pyt/lfads \
--device_num 0 \
--data_suffix ecog_fl0u10 \
--max_epochs 25000 \
--batch_size 10000 \
--seq_len 70 \
-t
