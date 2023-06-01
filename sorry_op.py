import os
import argparse

import torch

try:
    from src.dist import init_distributed_mode
except:
    from dist import init_distributed_mode

"""
inputs = torch.zeros(512, 512).to(0)
model = torch.nn.Linear(512, 512).to('cuda:0')

inputs_c = torch.zeros(512, 512)

while True:
    outputs = model(inputs)
    outputs_c = inputs_c - torch.ones_like(inputs_c)
    outputs = model(outputs)
    outputs.backward(torch.randn_like(outputs))
"""


def sorry_op(local_rank: int = None, feature_dim: int = 1024):

    if local_rank is None:
        try:
            local_rank = int(os.environ['LOCAL_RANK'])
        except Exception:
            local_rank = 0

    print(f'start sorry_op: local_rank: {local_rank}')

    inputs = torch.zeros(feature_dim, feature_dim).to(f'cuda:{local_rank}')
    inputs_c = torch.zeros(feature_dim, feature_dim).to(inputs.device)
    model = torch.nn.Linear(feature_dim, feature_dim).to(inputs.device)

    while True:
        outputs = model(inputs)
        outputs = model(outputs)
        outputs_c = inputs_c - torch.ones_like(inputs_c)
        outputs.backward(torch.randn_like(outputs))


def get_args_parser():
    p = argparse.ArgumentParser('', add_help=False)
    p.add_argument('--local_rank', type=int, default=0)
    p.add_argument('--world_size', type=int, default=1)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--dist_url', type=str, default='env://')
    p.add_argument('--dist_backend', type=str, default='nccl')
    p.add_argument('--ddp_timeout', default=1800, type=int, help='timeout in seconds (default=30min.)')
    p.add_argument('--distributed', type=bool, default=False)
    return p


if __name__ == '__main__':
    p = argparse.ArgumentParser('', parents=[get_args_parser()])
    opt = p.parse_args()

    init_distributed_mode(opt)

    # 8
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1234567 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20000 sorry_op.py
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=01234568 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20168 sorry_op.py
    # 7
    # CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 --nnodes=1 --rdzv_id=1234567 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20000 sorry_op.py
    # CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 --nnodes=1 --rdzv_id=01234568 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20168 sorry_op.py
    # 7
    # CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun --nproc_per_node=7 --nnodes=1 --rdzv_id=1234567 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20000 sorry_op.py
    # CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun --nproc_per_node=7 --nnodes=1 --rdzv_id=01234568 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20168 sorry_op.py
    # 7
    # CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun --nproc_per_node=7 --nnodes=1 --rdzv_id=1234567 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20000 sorry_op.py
    # CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun --nproc_per_node=7 --nnodes=1 --rdzv_id=01234568 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20168 sorry_op.py
    # 6
    # CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node=6 --nnodes=1 --rdzv_id=1234567 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20000 sorry_op.py
    # CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node=6 --nnodes=1 --rdzv_id=01234568 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20168 sorry_op.py
    # 6
    # CUDA_VISIBLE_DEVICES=0,2,3,4,5,7 torchrun --nproc_per_node=6 --nnodes=1 --rdzv_id=1234567 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20000 sorry_op.py
    # CUDA_VISIBLE_DEVICES=0,2,3,4,6,7 torchrun --nproc_per_node=6 --nnodes=1 --rdzv_id=01234568 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20168 sorry_op.py
    # 5 
    # CUDA_VISIBLE_DEVICES=3,4,5,6,7 torchrun --nproc_per_node=5 --nnodes=1 --rdzv_id=1234567 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20000 sorry_op.py
    # CUDA_VISIBLE_DEVICES=3,4,5,6,7 torchrun --nproc_per_node=5 --nnodes=1 --rdzv_id=01234568 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20168 sorry_op.py
    # 4
    # CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --nnodes=1 --rdzv_id=1234567 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20000 sorry_op.py
    # CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --nnodes=1 --rdzv_id=01234568 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20168 sorry_op.py
    # 4
    # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --nnodes=1 --rdzv_id=1234567 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20000 sorry_op.py
    # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --nnodes=1 --rdzv_id=01234568 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20168 sorry_op.py
    # 3
    # CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 --nnodes=1 --rdzv_id=1234567 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20000 sorry_op.py
    # CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 --nnodes=1 --rdzv_id=01234568 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20168 sorry_op.py
    # 2 
    # CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=1234567 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20000 sorry_op.py
    # CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=01234568 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20168 sorry_op.py
    # 3
    # CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 --nnodes=1 --rdzv_id=1234567 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20000 sorry_op.py
    # CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 --nnodes=1 --rdzv_id=01234568 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20168 sorry_op.py
    # 3
    # CUDA_VISIBLE_DEVICES=0,2,3 torchrun --nproc_per_node=3 --nnodes=1 --rdzv_id=1234567 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20000 sorry_op.py
    # CUDA_VISIBLE_DEVICES=0,2,3 torchrun --nproc_per_node=3 --nnodes=1 --rdzv_id=01234568 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20168 sorry_op.py
    # 3
    # CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --nnodes=1 --rdzv_id=1234567 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20000 sorry_op.py
    # CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --nnodes=1 --rdzv_id=01234568 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20168 sorry_op.py
    # 2 
    # CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=1234567 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20000 sorry_op.py
    # CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=01234568 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20168 sorry_op.py
    # 2
    # CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=01234569 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20169 sorry_op.py
    # CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=01234570 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20170 sorry_op.py
    # 2
    # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=01234569 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20169 sorry_op.py
    # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=01234570 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20170 sorry_op.py
    # 2
    # CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=01234569 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20169 sorry_op.py
    # CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=01234570 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20170 sorry_op.py
    # 2
    # CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=01234569 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20169 sorry_op.py
    # CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=01234570 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20170 sorry_op.py
    # 2
    # CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=01234569 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20169 sorry_op.py
    # CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=01234570 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20170 sorry_op.py
    # 2
    # CUDA_VISIBLE_DEVICES=1,7 torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=01234569 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20169 sorry_op.py
    # CUDA_VISIBLE_DEVICES=1,7 torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=01234570 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:20170 sorry_op.py
    print('sorry_op')
    sorry_op(opt.local_rank)
