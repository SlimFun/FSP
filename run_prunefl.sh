python3 prunefl.py --dataset cifar10 --rounds-between-readjustments 50 --initial-rounds 1000 --total-clients 10 --clients 10 --model VGG11_BN --rounds 100 --device 2 --batch-size 64 --l2 1e-5 --partition_method hetero --partition_alpha 0.5