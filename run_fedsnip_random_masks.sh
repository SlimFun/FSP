python3 -u fedsnip.py --dataset cifar10 --eta 0.01 --total-clients 10 --clients 10 --batch-size 64 --rounds 100 --model CNNNet --epochs 10  --device 3 --prune_strategy random_masks --keep_ratio 0.1 --target_keep_ratio 0.05 --prune_at_first_round --single_shot_pruning --partition_method hetero --partition_alpha 0.5