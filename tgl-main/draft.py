import argparse
import logging
import os
import time
import torch
import dgl
import datetime
import random
import math
import threading
import numpy as np
import torch.distributed as td


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--seed', type=int, default=0, help='random seed to use')
parser.add_argument('--num_gpus', type=int, default=4, help='number of gpus to use')
parser.add_argument('--omp_num_threads', type=int, default=8)
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--logging", type=str, default='warning')
parser.add_argument("--GLOO_SOCKET_IFNAME", type=str, default='')
args = parser.parse_args()



'''
sudo firewall-cmd --zone=public --remove-port=10000-65000/tcp --permanent
sudo firewall-cmd --zone=public --add-port=10000-65000/tcp --permanent
sudo firewall-cmd --reload
sudo firewall-cmd --zone=public --list-ports
'''

os.environ['NCCL_SOCKET_NPORTS'] = '11'
os.environ['NCCL_SOCKET_PORT_RANGE'] = '10000,10010'
os.environ['GLOO_SOCKET'] = '10000,10001,10002,10003,10004,10005,10006,10007,10008,10009,10010'

print("local_rank: ", args.local_rank)
td.init_process_group(backend='gloo', timeout=datetime.timedelta(0, 3600000), init_method='env://')
# td.init_process_group(backend='gloo', timeout=datetime.timedelta(0, 3600000), init_method='tcp://10.214.151.198:34567',
#                     rank=args.local_rank, world_size=2)

print("等待barrier1")
torch.distributed.barrier()
print("结束barrier1")

ranks = [0]
for i in range(torch.distributed.get_world_size() - 2):
    ranks.append(i + 2)
all_proc = torch.distributed.get_world_size() - 1
print("torch.distributed.get_world_size(): ", torch.distributed.get_world_size())
print(f'ranks: {ranks}, all_proc: {all_proc}')
nccl_group = torch.distributed.new_group(ranks=ranks, backend='nccl')

print("等待barrier2")
torch.distributed.barrier(group=nccl_group)
print("结束barrier2")



# model = GeneralModel(dim_feats[1], dim_feats[4], sample_param, memory_param, gnn_param, train_param).cuda()
# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], process_group=nccl_group,
#                                                   output_device=args.local_rank,
#                                                   find_unused_parameters=True)