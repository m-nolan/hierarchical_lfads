python train_model_ecog_reduced.py ^
-m lfads_ecog ^
-d D:\Users\mickey\Data\datasets\ecog\goose_wireless\gw_250_fl0u20 ^
-p C:\Users\mickey\aoLab\code\hierarchical_lfads\hyperparameters\ecog\lfads_ecog_3.yaml ^
-o D:\Users\mickey\Data\models\pyt\lfads ^
--data_suffix ecog_fl0u20 ^
--max_epochs 1500 ^
--batch_size 3000 ^
--seq_len 50 ^
-t