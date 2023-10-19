import logging

import torch
from torch import tensor
import torch.distributed.rpc as rpc



class KVstore:
    def __init__(self, node_feats: torch.Tensor, edge_feats: torch.Tensor, rank: int, world_size: int):
        self.node_feats = node_feats
        self.edge_feats = edge_feats

        self.dim_feats = [0, 0, 0, 0, 0, 0, False, False]
        if node_feats is not None:
            self.dim_feats[0] = node_feats.shape[0]
            self.dim_feats[1] = node_feats.shape[1]
            self.dim_feats[2] = node_feats.dtype
            self.dim_feats[6] = True
        if edge_feats is not None:
            self.dim_feats[3] = edge_feats.shape[0]
            self.dim_feats[4] = edge_feats.shape[1]
            self.dim_feats[5] = edge_feats.dtype
            self.dim_feats[7] = True

        self.node_feats_not_None = node_feats is not None
        self.edge_feats_not_None = edge_feats is not None

        self.rank = rank
        self.world_size = world_size

        rpc.init_rpc("worker%d" % rank, rank=rank, world_size=world_size,
                     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                         rpc_timeout=18000,
                         num_worker_threads=32,
                         _transports=["uv"],
                         _channels=["cma", "mpt_uv", "basic", "cuda_xth", "cuda_ipc", "cuda_basic"]
                     ))



    def pull_local(self, keys, mode):
        if mode == 'edge':
            return self.edge_feats[keys]
        elif mode == 'node':
            return self.node_feats[keys]
        else:
            raise ValueError('pull_local mode error')





global RANK
global KVSTORE
global ALL_KEYS
ALL_KEYS = 0
global PULL_KEYS
PULL_KEYS = 0

def acc_all_keys(x) -> int:
    global ALL_KEYS
    ALL_KEYS += x
    # print("all keys woc: ", ALL_KEYS)
    return ALL_KEYS

def acc_pull_keys(x) -> int:
    global PULL_KEYS
    PULL_KEYS += x
    # print("pull keys woc: ", PULL_KEYS)
    return PULL_KEYS

def set_rank(rank: int):
    global RANK
    RANK = rank

def get_rank() -> int:
    global RANK
    if RANK is None:
        raise RuntimeError("The rank has not been initialized.")
    return RANK

def get_kvstore() -> KVstore:
    global KVSTORE
    if KVSTORE is None:
        raise RuntimeError("The kvstore client has not been initialized.")
    return KVSTORE


def set_kvstore(kvstore: KVstore):
    global KVSTORE
    KVSTORE = kvstore


def pull_local(keys: torch.Tensor, mode: str) -> torch.Tensor:
    kvstore = get_kvstore()
    return kvstore.pull_local(keys, mode)


def pull_remote(keys, mode, rank) -> torch.Tensor:
    acc_all_keys(len(keys))
    # print("add wql ", len(keys))
    if rank == 0:
        return pull_local(keys, mode)
    else:
        acc_pull_keys(len(keys))
        return rpc.rpc_sync("worker0", pull_local, args=(keys, mode))