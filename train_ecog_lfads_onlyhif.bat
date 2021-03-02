python train_model_ecog_reduced.py ^
-m lfads_ecog ^
-d D:\Users\mickey\Data\datasets\ecog\goose_wireless\gw_250_fl50u120 ^
-p C:\Users\mickey\aoLab\code\hierarchical_lfads\hyperparameters\ecog\lfads_ecog_reduced_nofilt_allgpu.yaml ^
-o D:\Users\mickey\Data\models\pyt\lfads ^
--data_suffix ecog_fl50u120 ^
--max_epochs 1500 ^
--batch_size 1000 ^
--seq_len 50 ^
-t