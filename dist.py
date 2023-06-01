from typing import Tuple, List, Dict
import os

import numpy as np

import torch
import torch.distributed as dist

#NOTE: for ddp_timeout
import datetime


@torch.no_grad()
def broadcast_params(backbone: torch.nn.Module, src_rank: int = 0) -> None:
    for p in backbone.parameters():
        broadcast(p, src_rank) 


@torch.no_grad()
def broadcast_buffers(backbone: torch.nn.Module, src_rank: int = 0) -> None:
    for b in backbone.buffers():
        broadcast(b, src_rank) 


@torch.no_grad()
def broadcast_params_buffers(backbone: torch.nn.Module, src_rank: int = 0) -> None:
    broadcast_params(backbone, src_rank=src_rank)
    broadcast_buffers(backbone, src_rank=src_rank)


@torch.no_grad()
def broadcast(
    tensor: torch.Tensor,
    src_rank: int,
    group: int = None,
    async_op: bool = False
) -> None:
    assert not async_op, f'Boradcast funtion does not support async_op'

    if group is None:
        group = dist.group.WORLD 
    dist.broadcast(tensor, src_rank, group=group, async_op=async_op)


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


@torch.no_grad()
def all_reduce(
    tensor: torch.Tensor,
    op: str,
    group: int = None
) -> None:

    if op == 'sum': op = dist.ReduceOp.SUM
    elif op == 'prod': op = dist.ReduceOp.PRODUCT
    elif op == 'max': op = dist.ReduceOp.MAX
    elif op == 'min': op = dist.ReduceOp.MIN
    else: assert 0, f'Check op: {op} in all_reduce'

    if group is None: group = dist.group.WORLD
    dist.all_reduce(tensor, op=op, group=group)


@torch.no_grad()
def reduce_scatter(
    tensor: torch.Tensor,
    op: str, 
    dim: int = 0,
    world_size: int = None,
    rank: int = None,
    group: int = None,
    async_op: bool = False
) -> torch.Tensor:

    assert not async_op, \
        'reduce_scatter can not be performed asynchronously at this moment'

    if op == 'sum': op = dist.ReduceOp.SUM
    elif op == 'prod': op = dist.ReduceOp.PRODUCT
    elif op == 'max': op = dist.ReduceOp.MAX
    elif op == 'min': op = dist.ReduceOp.MIN
    else: assert 0, f'check op: {op} in reduce_scatter'

    if group is None:
        group = dist.group.WORLD

    if world_size is None: world_size = get_world_size()
    if rank is None: rank = get_rank()

    tensor_list = list(torch.chunk(tensor.data, world_size, dim=dim))
    output = torch.zeros_like(tensor_list[0])
    dist.reduce_scatter(output, tensor_list, op=op, group=group, async_op=async_op)

    return output


@torch.no_grad()
def gather(
    tensor: torch.Tensor,
    tensor_list: List[torch.Tensor] = None,
    root: int = 0,
    group: int = None
) -> None:

    """ Sends tensor to root process, which store it in tensor_list.
    """
    rank = dist.get_rank()
    if group is None:
        group = dist.group.WORLD
    if rank == root:
        assert(tensor_list is not None), f'tensor_list is None in gather'
        dist.gather(tensor.data, gather_list=tensor_list, group=group)
    else:
        dist.gather(tensor.data, dst=root, group=group)


@torch.no_grad()
def all_gather(
    tensor_list: List[torch.Tensor],
    tensor: torch.Tensor,
    group: int = None,
    async_op: bool = False
) -> None:
    assert not async_op, f'all_gather does not support async_op'

    if group is None:
        group = dist.group.WORLD

    dist.all_gather(tensor_list, tensor.data, group=group, async_op=async_op)


@torch.no_grad()
def make_output_list(
    tensor: torch.Tensor,
    requires_grad: bool = False
) -> torch.Tensor:
    world_size = get_world_size()
    return [
        torch.zeros_like(tensor, requires_grad=requires_grad) \
            for _ in range(world_size)
    ]


def postprocess_tensor_list(
    tensor_list: List[torch.Tensor],
    dim: int = None,
    requires_grad: bool = False,
    retain_grad: bool = False
) -> torch.Tensor:

    if dim is not None:
        tensor_list = torch.cat(tensor_list, dim=dim)
        tensor_list.requires_grad_(requires_grad)
        if retain_grad: tensor_list.retain_grad()
    else:
        for t in tensor_list:
            t.requires_grad_(requires_grad)
        if retain_grad:
            for t in tensor_list:
                t.retain_grad()

    return tensor_list


@torch.no_grad()
def all_gather_wo_tensor_list(
    tensor: torch.Tensor,
    group: int = None, 
    dim: int = None,
    requires_grad: bool = False,
    retain_grad: bool = False
) -> torch.Tensor:

    if group is None: group = dist.group.WORLD

    tensor_list = make_output_list(tensor)
    dist.all_gather(tensor_list, tensor.data, group=group)
    tensor_list = postprocess_tensor_list(
        tensor_list, dim=dim,
        requires_grad=requires_grad, retain_grad=retain_grad
    )

    return tensor_list


@torch.no_grad()
def all_gather_wo_tensor_list_async(
    tensor: torch.Tensor,
    group: int = None
) -> Tuple[torch.Tensor, object]:

    if group is None:
        group = dist.group.WORLD

    tensor_list = make_output_list(tensor)
    async_handler = dist.all_gather(
        tensor_list,
        tensor.data,
        group=group,
        async_op=True
    )

    return tensor_list, async_handler


def synchronize_distributed_collectives(
    stream: object,
    handler: object,
    variable: torch.Tensor
) -> torch.Tensor:
    # https://pytorch.org/docs/stable/distributed.html

    # handler.wait() ensures the operation is enqueued, but not necessarily complete.
    handler.wait()
    # Using result on non-default stream.
    with torch.cuda.stream(stream):
        stream.wait_stream(torch.cuda.default_stream()) 
        return variable

    assert 0, \
        f'fail to synchronize_distributed_collectives({stream}, {handler})'


def is_main_process() -> bool:
    return get_rank() == 0


def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_process_group() -> object:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.group.WORLD


def get_grad_bucket():
    return dist._GradBucket


def init_distributed_mode(args: Dict) -> None:

    # If already distributed mode is activated
    if args.distributed: return

    try:
        args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.group_rank = int(os.environ['GROUP_RANK'])
        args.role_rank = int(os.environ['ROLE_RANK'])
        args.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        args.role_world_size = int(os.environ['ROLE_WORLD_SIZE'])
        args.master_addr = os.environ['MASTER_ADDR']
        args.master_port = os.environ['MASTER_PORT']
        args.torchelastic_restart_count = os.environ['TORCHELASTIC_RESTART_COUNT']
        args.torchelastic_max_restarts = os.environ['TORCHELASTIC_MAX_RESTARTS']
        args.torchelastic_run_id = os.environ['TORCHELASTIC_RUN_ID']
        args.omp_num_threads = os.environ['OMP_NUM_THREADS']
    except:
        print(f'In Non-distributed mode')
        args.distributed = False
        setup_for_distributed(args.rank==0)
        return

    # NOTE: Set distributed-flag dynamically at first
    args.distributed = True

    # Set default gpu-card
    torch.cuda.set_device(args.local_rank)

    # Printing Elastic Launch Info., and then Init.
    print(f'torch.distributed.init_process_group: '\
          f'world_size {args.world_size}, '\
          f'rank {args.rank}, '\
          f'local_rank: {args.local_rank}, '\
          f'dist_url: {args.dist_url}, '\
          f'torchelastic_run_id: {args.torchelastic_run_id}', flush=True)
    print(f'group_rank: {args.group_rank}, '\
          f'role_rank: {args.role_rank}, '\
          f'local_world_size: {args.local_world_size}, '\
          f'role_world_size: {args.role_world_size}, '\
          f'master_addr: {args.master_addr}, '\
          f'master_port: {args.master_port}, '\
          f'torchelastic_restart_count: {args.torchelastic_restart_count}, '\
          f'torchelastic_max_restarts: {args.torchelastic_max_restarts}, '\
          f'omp_num_threads: {args.omp_num_threads}',
          flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(seconds=args.ddp_timeout)
    )
    torch.distributed.barrier()

    if torch.distributed.is_initialized():
        if is_main_process():
            print(f'torch.distributed.is_torchelastic_launched(): '\
                  f'{torch.distributed.is_torchelastic_launched()}')
            print('Congratulations!!! torch.distributed is launched successfully!!!')
    else:
        print('Fail to torch.distributed.init_process_group(). exit(-1)')
        exit(-1)

    # Console-flushing will be conducted from the rank0 process only
    # after executing the setup_for_distributed
    setup_for_distributed(args.rank==0)


def dist_barrier() -> None:
    if is_dist_avail_and_initialized():
        torch.distributed.barrier()
    else: pass


def destory_distributed_mode() -> bool:
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    print('Done, torch.distributed.destroy_process_group()')
    return True


def setup_for_distributed(is_master: int) -> None:
    # This function disables to console-printing when not in master process (i.e. rank=0)
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def ddp(
    module: torch.nn.Module,
    local_rank: int,
    broadcast_buffers: bool = False,
    find_unused_parameters: bool = False
):
    device_ids = [local_rank]
    output_device = [local_rank]
    #NOTE
    # broadcast_buffers (bool) – Flag that enables syncing (broadcasting) buffers of 
    # the module at beginning of the forward function. (default: True)
    # It is useful in case that a pytorch-module in your model has a buffer variables(e.g. see BatchNorm)
    module = torch.nn.parallel.DistributedDataParallel(
                 module=module, 
                 device_ids=device_ids,
                 output_device=output_device,
                 broadcast_buffers=broadcast_buffers,
                 find_unused_parameters=find_unused_parameters,
    )
    return module


def ddp_compress_hook(modules_list: List[torch.nn.Module]):
    if torch.cuda.is_bf16_supported():
        # NOTE: bf16 rather than fp16
        # https://moocaholic.medium.com/fp64-fp32-fp16-bfloat16-tf32-and-other-members-of-the-zoo-a1ca7897d407
        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook
        for module in modules_list:
            module.register_comm_hook(get_process_group(), bf16_compress_hook)
        print(f'bf16_compress_hook activated')
    else:
        # In case, TypeError: BF16 all reduce communication hook required CUDA 11+, NCCL 2.9.7+, and A100+.
        # Rollback to the fp16_compress_hook
        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
        for module in modules_list:
            module.register_comm_hook(get_process_group(), fp16_compress_hook)
        print(f'fp16_compress_hook activated')
    return True


def all_gather_for_non_shuffled_dataloader(
    features: torch.Tensor,
    world_size: int
) -> torch.Tensor:
    #NOTE: features: B x D
    _features_gathered = all_gather_wo_tensor_list(features, dim=None)
    batch_size_per_rank = \
        np.array(
            [batch_data.shape[0] for batch_data in _features_gathered]
        )
    features_gathered = torch.empty(
        batch_size_per_rank.sum(),
        features.shape[1],
        device=_features_gathered[0].device
    )
    for r in range(world_size):
        features_gathered[r::world_size] = _features_gathered[r]

    return features_gathered
