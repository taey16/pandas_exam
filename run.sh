CUDA_VISIBLE_DEVICES=0 nohup python main.py > log_device0.log &
CUDA_VISIBLE_DEVICES=1 nohup python main.py > log_device1.log &

CUDA_VISIBLE_DEVICES=0 python main.py
CUDA_VISIBLE_DEVICES=1 python main.py
