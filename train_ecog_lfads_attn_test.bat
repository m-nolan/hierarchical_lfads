python train_model_ecog_reduced.py ^
-m lfads_ecog ^
-d D:\Users\mickey\Data\datasets\ecog\goose_wireless\gw_250 ^
-p C:\Users\mickey\aoLab\code\hierarchical_lfads\hyperparameters\ecog\lfads_ecog_4.yaml ^
-o C:\Users\mickey\aoLab\code\hierarchical_lfads\asdf ^
--data_suffix ecog ^
--max_epochs 200 ^
--batch_size 1000 ^
--seq_len 50 ^
-t