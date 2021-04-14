#!/bin/bash

python train_model_ecog_reduced.py \
-m lfads_ecog \
-d /home/ws5/manolan/data/datasets/ecog/goose_wireless/gw_250_fl50u120 \
-p /home/ws5/manolan/ecog_pred/hierarchical_lfads/hyperparameters/ecog/lfads_ecog_reduced_nofilt_allgpu.yaml \
-o /home/ws5/manolan/data/models/pyt/lfads \
--data_suffix ecog_fl50u120 \
--max_epochs 2500 \
--batch_size 1500 \
--seq_len 50 \
-t