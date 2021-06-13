python train_model_ecog_reduced.py ^
-m lfads_ecog ^
-d D:\Users\mickey\Data\datasets\ecog\goose_wireless\gw_250_renorm ^
-p D:\Users\mickey\aoLab\code\hierarchical_lfads\hyperparameters\ecog\lfads_ecog_reduced_nofilt_allgpu.yaml ^
-o D:\Users\mickey\Data\models\pyt\lfads ^
--data_suffix ecog ^
--max_epochs 2500 ^
--batch_size 500 ^
-t