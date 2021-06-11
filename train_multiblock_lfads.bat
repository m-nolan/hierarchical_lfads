python train_model_multiblock_ecog.py ^
-m multiblock_lfads_ecog ^
-d D:\Users\mickey\Data\datasets\ecog\goose_wireless\gw_250_renorm ^
-p D:\Users\mickey\aoLab\code\hierarchical_lfads\hyperparameters\ecog\lfads_ecog_recon_multiblock.yaml ^
-o D:\Users\mickey\Data\models\pyt\lfads ^
--data_suffix ecog ^
--max_epochs 1500 ^
--batch_size 500 ^
-t