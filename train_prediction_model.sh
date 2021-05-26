#!/bin/bash


python train_model_ecog_prediction.py \
-m lfads_ecog \
-d /home/ws5/manolan/data/datasets/ecog/goose_wireless/gw_250 \
-p /home/ws5/manolan/ecog_pred/hierarchical_lfads/hyperparameters/ecog/lfads_ecog_reduced_nofilt_allgpu.yaml \
-o /home/ws5/manolan/data/models/pyt/lfads \
--data_suffix ecog \
--max_epochs 1500 \
--batch_size 1000 \
--device 1 \
-t