python3 -u fedsnip.py --dataset cifar10 --eta 0.01 --distribution classic_iid --total-clients 10 --clients 10 --batch-size 32 --rounds 100 --model CIFAR10Net --epochs 10  --device 2 --prune_strategy SNIP --keep_ratio 0.1 --prune_at_first_round --single_shot_pruning --l2 0.001
