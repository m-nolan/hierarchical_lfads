python train_model_filtnoise_sweep.py ^
-m lfads_ecog ^
-p D:\Users\mickey\aoLab\code\hierarchical_lfads\hyperparameters\ecog\lfads_ecog_reduced_nofilt_allgpu.yaml ^
-o D:\Users\mickey\Data\models\pyt\lfads ^
--max_epochs 300 ^
--batch_size 500 ^
--seq_len 50 ^
-t