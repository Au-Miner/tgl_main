import argparse
import os
import time


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--seed', type=int, default=0, help='random seed to use')
parser.add_argument('--num_gpus', type=int, default=4, help='number of gpus to use')
parser.add_argument('--omp_num_threads', type=int, default=8)
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

'''
1. node_features.pt
2. edge_features.pt
3. edges.csv
4. ext_full.npz————need to be generated

//python -m torch.distributed.launch --nproc_per_node=2 train_dist1.py --data WIKI_0 --config config/TGN.yml --num_gpus 1

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="10.214.151.191" --master_port=34567 train_dist2.py --data WIKI --config config/TGN.yml --num_gpus=1

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="10.214.151.191" --master_port=34567 train_dist2.py --data WIKI --config config/TGN.yml --num_gpus=1

rsync -avz /home/qcsun/wql_tgl/tgl-main qcsun@node192:/home/qcsun/wql_tgl

双机tgl原版
需要要做的内容：
所有机子有一个cpu进程和一个gpu进程，cpu负责采样
直接把模型套到ddp上，就可以分布式采样了，不需要一个主cpu进程统筹管理模型调度
第一台机子负责处理前50%数据，第二台机子负责后50%数据
'''

# set which GPU to use
if args.local_rank < args.num_gpus:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.local_rank)
    print("os.environ['CUDA_VISIBLE_DEVICES']: ", os.environ['CUDA_VISIBLE_DEVICES'])
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print("os.environ['CUDA_VISIBLE_DEVICES']: ", os.environ['CUDA_VISIBLE_DEVICES'])
os.environ['OMP_NUM_THREADS'] = str(args.omp_num_threads)
os.environ['MKL_NUM_THREADS'] = str(args.omp_num_threads)



import torch
import dgl
import datetime
import random
import math
import threading
import numpy as np
from tqdm import tqdm
from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
from sklearn.metrics import average_precision_score, roc_auc_score
from modules import *
from sampler import *
from utils import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

all_total_time = 0
all_sample_time = 0
all_train_time = 0
lis_total_time = []
lis_sample_time = []
lis_train_time = []

set_seed(args.seed)
torch.distributed.init_process_group(backend='gloo', timeout=datetime.timedelta(0, 3600), init_method='env://')
# 注意这里group定义
ranks = [0, 2, 3, 4]
all_proc = 4
# nccl_group = torch.distributed.new_group(ranks=list(range(args.num_gpus)), backend='nccl')
nccl_group = torch.distributed.new_group(ranks=ranks, backend='nccl')

# local_rank范围从[0, num_gpus]，其中num_gpus为cpu执行
if args.local_rank == 0:
    # 加载点特征和边特征
    _node_feats, _edge_feats = load_feat(args.data)
dim_feats = [0, 0, 0, 0, 0, 0]
# 对于第1个GPU创建共享变量
if args.local_rank == 0:
    # print("准备开始读取节点特征和边特征")
    if _node_feats is not None:
        # 创建内存共享节点/边特征变量node_feats和edge_feats
        # dim_feats[0]表示节点个数，dim_feats[1]表示节点dim，dim_feats[2]表示节点类型
        dim_feats[0] = _node_feats.shape[0]
        dim_feats[1] = _node_feats.shape[1]
        dim_feats[2] = _node_feats.dtype
        node_feats = create_shared_mem_array('node_feats', _node_feats.shape, dtype=_node_feats.dtype)
        # node_feats = create_shared_mem_array('node_feats', _node_feats.shape, dtype=torch.float32)
        node_feats.copy_(_node_feats)
        # print("正在读取节点特征，节点特征总共有", _node_feats.size())
        del _node_feats
    else:
        node_feats = None
    if _edge_feats is not None:
        # print("woc: ", _edge_feats.dtype)
        # print("woc: ", _edge_feats.shape[0])
        # print("woc: ", _edge_feats.shape[1])
        # print("woc: ", _edge_feats.dtype)
        # print(args.local_rank, "===111")
        dim_feats[3] = _edge_feats.shape[0]
        dim_feats[4] = _edge_feats.shape[1]
        dim_feats[5] = _edge_feats.dtype
        edge_feats = create_shared_mem_array('edge_feats', _edge_feats.shape, dtype=_edge_feats.dtype)
        # edge_feats = create_shared_mem_array('edge_feats', _edge_feats.shape, dtype=torch.float32)
        # print(args.local_rank, "===222")
        edge_feats.copy_(_edge_feats)
        # print("正在读取边特征，边特征总共有", _edge_feats.size())
        del _edge_feats
    else:
        edge_feats = None
# 进程第一次同步，保证edge_feats和node_feats被移动到内存中
# print(args.local_rank, "===333")
torch.distributed.barrier()
# print(args.local_rank, "===444")
torch.distributed.broadcast_object_list(dim_feats, src=0)
# print(args.local_rank, "===555")
# 其他gpu进程从内存中读取edge_feats和node_feats
if args.local_rank > 0 and args.local_rank < args.num_gpus:
    node_feats = None
    edge_feats = None
    if os.path.exists('/home/qcsun/DATA/{}/node_features.pt'.format(args.data)):
        node_feats = get_shared_mem_array('node_feats', (dim_feats[0], dim_feats[1]), dtype=dim_feats[2])
    if os.path.exists('/home/qcsun/DATA/{}/edge_features.pt'.format(args.data)):
        edge_feats = get_shared_mem_array('edge_feats', (dim_feats[3], dim_feats[4]), dtype=dim_feats[5])
sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
orig_batch_size = train_param['batch_size']
# 定义模型存储路径
if args.local_rank == 0:
    if not os.path.isdir('models'):
        os.mkdir('models')
    path_saver = ['models/{}_{}.pkl'.format(args.data, time.time())]
else:
    path_saver = [None]
torch.distributed.broadcast_object_list(path_saver, src=0)
path_saver = path_saver[0]

print("dim_node", dim_feats[1])
print("dim_node", dim_feats[1])
print("dim_node", dim_feats[1])
print("dim_node", dim_feats[1])

# 如果是最后一个进程，即CPU进程，则获取图信息g和所有边信息df
if args.local_rank == args.num_gpus:
    g, df = load_graph(args.data)
    num_nodes = [g['indptr'].shape[0] - 1]
else:
    num_nodes = [None]
# 等待cpu读取完g和df，然后将节点个数广播给所有进程
torch.distributed.barrier()
torch.distributed.broadcast_object_list(num_nodes, src=args.num_gpus)
num_nodes = num_nodes[0]

mailbox = None
# 如果需要memory，则创建mailbox的内存共享环境，mailbox可以理解为就是用来存memory的数据结构
if memory_param['type'] != 'none':
    if args.local_rank == 0:
        node_memory = create_shared_mem_array('node_memory', torch.Size([num_nodes, memory_param['dim_out']]),
                                              dtype=torch.float32)
        node_memory_ts = create_shared_mem_array('node_memory_ts', torch.Size([num_nodes]), dtype=torch.float32)
        mails = create_shared_mem_array('mails', torch.Size(
            [num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_feats[4]]), dtype=torch.float32)
        mail_ts = create_shared_mem_array('mail_ts', torch.Size([num_nodes, memory_param['mailbox_size']]),
                                          dtype=torch.float32)
        next_mail_pos = create_shared_mem_array('next_mail_pos', torch.Size([num_nodes]), dtype=torch.long)
        update_mail_pos = create_shared_mem_array('update_mail_pos', torch.Size([num_nodes]), dtype=torch.int32)
        torch.distributed.barrier()
        node_memory.zero_()
        node_memory_ts.zero_()
        mails.zero_()
        mail_ts.zero_()
        next_mail_pos.zero_()
        update_mail_pos.zero_()
    else:
        torch.distributed.barrier()
        node_memory = get_shared_mem_array('node_memory', torch.Size([num_nodes, memory_param['dim_out']]),
                                           dtype=torch.float32)
        node_memory_ts = get_shared_mem_array('node_memory_ts', torch.Size([num_nodes]), dtype=torch.float32)
        mails = get_shared_mem_array('mails', torch.Size(
            [num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_feats[4]]), dtype=torch.float32)
        mail_ts = get_shared_mem_array('mail_ts', torch.Size([num_nodes, memory_param['mailbox_size']]),
                                       dtype=torch.float32)
        next_mail_pos = get_shared_mem_array('next_mail_pos', torch.Size([num_nodes]), dtype=torch.long)
        update_mail_pos = get_shared_mem_array('update_mail_pos', torch.Size([num_nodes]), dtype=torch.int32)
    mailbox = MailBox(memory_param, num_nodes, dim_feats[4], node_memory, node_memory_ts, mails, mail_ts, next_mail_pos,
                      update_mail_pos)


# 数据管道线程————暂时不知道该类作用
class DataPipelineThread(threading.Thread):

    def __init__(self, my_mfgs, my_root, my_ts, my_eid, my_block, stream):
        super(DataPipelineThread, self).__init__()
        self.my_mfgs = my_mfgs
        self.my_root = my_root
        self.my_ts = my_ts
        self.my_eid = my_eid
        self.my_block = my_block
        self.stream = stream
        self.mfgs = None
        self.root = None
        self.ts = None
        self.eid = None
        self.block = None

    def run(self):
        with torch.cuda.stream(self.stream):
            # print(args.local_rank, 'start thread')
            nids, eids = get_ids(self.my_mfgs[0], node_feats, edge_feats)
            mfgs = mfgs_to_cuda(self.my_mfgs[0])
            # print("222mfgs为", mfgs)
            prepare_input(mfgs, node_feats, edge_feats, pinned=True, nfeat_buffs=pinned_nfeat_buffs,
                          efeat_buffs=pinned_efeat_buffs, nids=nids, eids=eids)
            # print("333mfgs为", mfgs)
            self.mfgs = mfgs
            if mailbox is not None:
                # print("此时mailbox非空")
                mailbox.prep_input_mails(mfgs[0], use_pinned_buffers=True)
                self.mfgs = mfgs
                # print("444mfgs为", mfgs)
                self.root = self.my_root[0]
                self.ts = self.my_ts[0]
                self.eid = self.my_eid[0]
                if memory_param['deliver_to'] == 'neighbors':
                    self.block = self.my_block[0]
            # print(args.local_rank, 'finished')

    def get_stream(self):
        return self.stream

    def get_mfgs(self):
        return self.mfgs

    def get_root(self):
        return self.root

    def get_ts(self):
        return self.ts

    def get_eid(self):
        return self.eid

    def get_block(self):
        return self.block


# 如果是gpu进程
if args.local_rank < args.num_gpus:
    # 创建模型
    model = GeneralModel(dim_feats[1], dim_feats[4], sample_param, memory_param, gnn_param, train_param).cuda()
    find_unused_parameters = True if sample_param['history'] > 1 else False
    # 将其设置为分布式模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], process_group=nccl_group,
                                                      output_device=args.local_rank,
                                                      find_unused_parameters=find_unused_parameters)
    # 创建损失函数
    creterion = torch.nn.BCEWithLogitsLoss()
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
    # 定义采样点/边固定缓冲区（大小为[n/e limit, n/e dim]）
    pinned_nfeat_buffs, pinned_efeat_buffs = get_pinned_buffers(sample_param, train_param['batch_size'], node_feats,
                                                                edge_feats)
    if mailbox is not None:
        mailbox.allocate_pinned_memory_buffers(sample_param, train_param['batch_size'])
    tot_loss = 0
    prev_thread = None
    while True:
        my_model_state = [None]
        # model_state = [None] * (args.num_gpus + 1)
        model_state = [None] * (all_proc + 1)
        # 获取当前模型状态，因为非cpu进程，故model_state可以设置为None
        torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
        if my_model_state[0] == -1:
            break
        elif my_model_state[0] == 4:
            continue
        # 如果状态为2，则将模型信息下载到本地
        elif my_model_state[0] == 2:
            torch.save(model.state_dict(), path_saver)
            continue
        # 如果状态为3，则将从本地下载模型信息
        elif my_model_state[0] == 3:
            model.load_state_dict(torch.load(path_saver, map_location=torch.device('cuda:0')))
            continue
        # 如果状态为5，将损失值发送给cpu进程，因为是发送方，所以第二个参数为None
        elif my_model_state[0] == 5:
            torch.distributed.gather_object(float(tot_loss), None, dst=args.num_gpus)
            tot_loss = 0
            continue
        # 如果状态为0，则gpu开始训练
        elif my_model_state[0] == 0:
            # 如果已经正在读数据了
            if prev_thread is not None:
                my_mfgs = [None]
                # multi_mfgs = [None] * (args.num_gpus + 1)
                multi_mfgs = [None] * (all_proc + 1)
                my_root = [None]
                # multi_root = [None] * (args.num_gpus + 1)
                multi_root = [None] * (all_proc + 1)
                my_ts = [None]
                # multi_ts = [None] * (args.num_gpus + 1)
                multi_ts = [None] * (all_proc + 1)
                my_eid = [None]
                # multi_eid = [None] * (args.num_gpus + 1)
                multi_eid = [None] * (all_proc + 1)
                my_block = [None]
                # multi_block = [None] * (args.num_gpus + 1)
                multi_block = [None] * (all_proc + 1)
                torch.distributed.scatter_object_list(my_mfgs, multi_mfgs, src=args.num_gpus)
                # print('1接收到数据，开始处理')
                if mailbox is not None:
                    torch.distributed.scatter_object_list(my_root, multi_root, src=args.num_gpus)
                    torch.distributed.scatter_object_list(my_ts, multi_ts, src=args.num_gpus)
                    torch.distributed.scatter_object_list(my_eid, multi_eid, src=args.num_gpus)
                    if memory_param['deliver_to'] == 'neighbors':
                        torch.distributed.scatter_object_list(my_block, multi_block, src=args.num_gpus)
                # print('1准备返回时间')
                stream = torch.cuda.Stream()
                # 创建当前线程，用来读取新一轮的数据
                # print("创建新线程")
                curr_thread = DataPipelineThread(my_mfgs, my_root, my_ts, my_eid, my_block, stream)
                curr_thread.start()
                # 让之前线程优先执行，等待之前线程执行完毕
                prev_thread.join()
                # with torch.cuda.stream(prev_thread.get_stream()):
                # 获取之前线程的读取结果mfgs
                mfgs = prev_thread.get_mfgs()
                # print("woc: ", mfgs)
                # print("555mfgs为", mfgs)
                # 进入训练模式开始训练
                # time_train_tmp = time.time()
                model.train()
                # print(111)
                optimizer.zero_grad()
                # print(222)
                # print("woc2: ", mfgs)
                pred_pos, pred_neg = model(mfgs)
                # print(333)
                loss = creterion(pred_pos, torch.ones_like(pred_pos))
                loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                # print(444)
                loss.backward()
                optimizer.step()
                # print(555)
                # time_train = time.time() - time_train_tmp
                # torch.distributed.gather_object(float(time_train), None, dst=args.num_gpus)
                with torch.no_grad():
                    tot_loss += float(loss)
                # 训练完之后更新一下mailbox和memory
                # print(666)
                if mailbox is not None:
                    with torch.no_grad():
                        eid = prev_thread.get_eid()
                        mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                        root_nodes = prev_thread.get_root()
                        ts = prev_thread.get_ts()
                        block = prev_thread.get_block()
                        mailbox.update_mailbox(model.module.memory_updater.last_updated_nid,
                                               model.module.memory_updater.last_updated_memory, root_nodes, ts,
                                               mem_edge_feats, block)
                        mailbox.update_memory(model.module.memory_updater.last_updated_nid,
                                              model.module.memory_updater.last_updated_memory,
                                              root_nodes,
                                              model.module.memory_updater.last_updated_ts)
                        # print(888)
                        if memory_param['deliver_to'] == 'neighbors':
                            # print(999)
                            torch.distributed.barrier(group=nccl_group)
                            # print(101010)
                            if args.local_rank == 0:
                                # print(111111)
                                mailbox.update_next_mail_pos()
                                # print(121212)
                # print(999)
                prev_thread = curr_thread
            else:
                my_mfgs = [None]
                # multi_mfgs = [None] * (args.num_gpus + 1)
                multi_mfgs = [None] * (all_proc + 1)
                my_root = [None]
                # multi_root = [None] * (args.num_gpus + 1)
                multi_root = [None] * (all_proc + 1)
                my_ts = [None]
                # multi_ts = [None] * (args.num_gpus + 1)
                multi_ts = [None] * (all_proc + 1)
                my_eid = [None]
                # multi_eid = [None] * (args.num_gpus + 1)
                multi_eid = [None] * (all_proc + 1)
                my_block = [None]
                # multi_block = [None] * (args.num_gpus + 1)
                multi_block = [None] * (all_proc + 1)
                # 获取mfgs
                torch.distributed.scatter_object_list(my_mfgs, multi_mfgs, src=args.num_gpus)
                # print('111接收到数据，开始处理', my_mfgs)
                if mailbox is not None:
                    torch.distributed.scatter_object_list(my_root, multi_root, src=args.num_gpus)
                    torch.distributed.scatter_object_list(my_ts, multi_ts, src=args.num_gpus)
                    torch.distributed.scatter_object_list(my_eid, multi_eid, src=args.num_gpus)
                    if memory_param['deliver_to'] == 'neighbors':
                        torch.distributed.scatter_object_list(my_block, multi_block, src=args.num_gpus)
                # print('2准备返回时间')
                stream = torch.cuda.Stream()
                # 创建之前线程
                # print("创建之前线程")
                # print("666: ", my_mfgs)
                # print("666: ", my_root)
                # print("666: ", my_ts)
                # print("666: ", my_eid)
                # print("666: ", my_block)
                # print("666: ", stream)
                prev_thread = DataPipelineThread(my_mfgs, my_root, my_ts, my_eid, my_block, stream)
                prev_thread.start()
        # 如果状态为1，则进行交叉验证
        elif my_model_state[0] == 1:
            if prev_thread is not None:
                # 完成最后的小批量训练，和状态为0类似
                prev_thread.join()
                mfgs = prev_thread.get_mfgs()
                model.train()
                optimizer.zero_grad()
                pred_pos, pred_neg = model(mfgs)
                loss = creterion(pred_pos, torch.ones_like(pred_pos))
                loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    tot_loss += float(loss)
                if mailbox is not None:
                    with torch.no_grad():
                        eid = prev_thread.get_eid()
                        mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                        root_nodes = prev_thread.get_root()
                        ts = prev_thread.get_ts()
                        block = prev_thread.get_block()
                        mailbox.update_mailbox(model.module.memory_updater.last_updated_nid,
                                               model.module.memory_updater.last_updated_memory, root_nodes, ts,
                                               mem_edge_feats, block)
                        mailbox.update_memory(model.module.memory_updater.last_updated_nid,
                                              model.module.memory_updater.last_updated_memory,
                                              root_nodes,
                                              model.module.memory_updater.last_updated_ts)
                        if memory_param['deliver_to'] == 'neighbors':
                            torch.distributed.barrier(group=nccl_group)
                            if args.local_rank == 0:
                                mailbox.update_next_mail_pos()
                prev_thread = None
            # 不开额外线程了，直接开始运行
            my_mfgs = [None]
            # multi_mfgs = [None] * (args.num_gpus + 1)
            multi_mfgs = [None] * (all_proc + 1)
            torch.distributed.scatter_object_list(my_mfgs, multi_mfgs, src=args.num_gpus)
            mfgs = mfgs_to_cuda(my_mfgs[0])
            prepare_input(mfgs, node_feats, edge_feats, pinned=True, nfeat_buffs=pinned_nfeat_buffs,
                          efeat_buffs=pinned_efeat_buffs)
            model.eval()
            with torch.no_grad():
                if mailbox is not None:
                    mailbox.prep_input_mails(mfgs[0])
                pred_pos, pred_neg = model(mfgs)
                if mailbox is not None:
                    my_root = [None]
                    # multi_root = [None] * (args.num_gpus + 1)
                    multi_root = [None] * (all_proc + 1)
                    my_ts = [None]
                    # multi_ts = [None] * (args.num_gpus + 1)
                    multi_ts = [None] * (all_proc + 1)
                    my_eid = [None]
                    # multi_eid = [None] * (args.num_gpus + 1)
                    multi_eid = [None] * (all_proc + 1)
                    torch.distributed.scatter_object_list(my_root, multi_root, src=args.num_gpus)
                    torch.distributed.scatter_object_list(my_ts, multi_ts, src=args.num_gpus)
                    torch.distributed.scatter_object_list(my_eid, multi_eid, src=args.num_gpus)
                    eid = my_eid[0]
                    mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                    root_nodes = my_root[0]
                    ts = my_ts[0]
                    block = None
                    if memory_param['deliver_to'] == 'neighbors':
                        my_block = [None]
                        # multi_block = [None] * (args.num_gpus + 1)
                        multi_block = [None] * (all_proc + 1)
                        torch.distributed.scatter_object_list(my_block, multi_block, src=args.num_gpus)
                        block = my_block[0]
                    mailbox.update_mailbox(model.module.memory_updater.last_updated_nid,
                                           model.module.memory_updater.last_updated_memory, root_nodes, ts,
                                           mem_edge_feats, block)
                    mailbox.update_memory(model.module.memory_updater.last_updated_nid,
                                          model.module.memory_updater.last_updated_memory,
                                          root_nodes,
                                          model.module.memory_updater.last_updated_ts)
                    if memory_param['deliver_to'] == 'neighbors':
                        torch.distributed.barrier(group=nccl_group)
                        if args.local_rank == 0:
                            mailbox.update_next_mail_pos()
                y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
                y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
                ap = average_precision_score(y_true, y_pred)
                auc = roc_auc_score(y_true, y_pred)
                torch.distributed.gather_object(float(ap), None, dst=args.num_gpus)
                torch.distributed.gather_object(float(auc), None, dst=args.num_gpus)
# 如果是cpu进程，即主进程
else:
    # 获取训练集和交叉验证集的终止位置
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]
    sampler = None
    # 定义正负采样对象
    if not ('no_sample' in sample_param and sample_param['no_sample']):
        sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                  sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                                  sample_param['strategy'] == 'recent', sample_param['prop_time'],
                                  sample_param['history'], float(sample_param['duration']))
    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)


    # 进行交叉验证和测试工作
    def eval(mode='val'):
        if mode == 'val':
            eval_df = df[train_edge_end:val_edge_end]
        elif mode == 'test':
            eval_df = df[val_edge_end:]
        elif mode == 'train':
            eval_df = df[:train_edge_end]
        # 主要实现步骤与该函数后面的代码类似，这里不细加注释
        ap_tot = list()
        auc_tot = list()
        train_param['batch_size'] = orig_batch_size
        # itr_tot = max(len(eval_df) // train_param['batch_size'] // args.num_gpus, 1) * args.num_gpus
        itr_tot = max(len(eval_df) // train_param['batch_size'] // all_proc, 1) * all_proc
        train_param['batch_size'] = math.ceil(len(eval_df) / itr_tot)
        multi_mfgs = list()
        multi_root = list()
        multi_ts = list()
        multi_eid = list()
        multi_block = list()
        for _, rows in eval_df.groupby(eval_df.index // train_param['batch_size']):
            root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(
                np.int32)
            ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = root_nodes.shape[0] * 2 // 3
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'], cuda=False)
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=False)
            multi_mfgs.append(mfgs)
            multi_root.append(root_nodes)
            multi_ts.append(ts)
            multi_eid.append(rows['Unnamed: 0'].values)
            if mailbox is not None and memory_param['deliver_to'] == 'neighbors':
                multi_block.append(to_dgl_blocks(ret, sample_param['history'], reverse=True, cuda=False)[0][0])
            # if len(multi_mfgs) == args.num_gpus:
            if len(multi_mfgs) == all_proc:
                # 修改状态为1，进行交叉验证
                # model_state = [1] * (args.num_gpus + 1)
                model_state = [1] * (all_proc + 1)
                my_model_state = [None]
                torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
                # multi_mfgs.append(None)
                multi_mfgs.insert(1, None)
                my_mfgs = [None]
                torch.distributed.scatter_object_list(my_mfgs, multi_mfgs, src=args.num_gpus)
                if mailbox is not None:
                    # multi_root.append(None)
                    multi_root.insert(1, None)
                    # multi_ts.append(None)
                    multi_ts.insert(1, None)
                    # multi_eid.append(None)
                    multi_eid.insert(1, None)
                    my_root = [None]
                    my_ts = [None]
                    my_eid = [None]
                    torch.distributed.scatter_object_list(my_root, multi_root, src=args.num_gpus)
                    torch.distributed.scatter_object_list(my_ts, multi_ts, src=args.num_gpus)
                    torch.distributed.scatter_object_list(my_eid, multi_eid, src=args.num_gpus)
                    if memory_param['deliver_to'] == 'neighbors':
                        # multi_block.append(None)
                        multi_block.insert(1, None)
                        my_block = [None]
                        torch.distributed.scatter_object_list(my_block, multi_block, src=args.num_gpus)
                # gathered_ap = [None] * (args.num_gpus + 1)
                gathered_ap = [None] * (all_proc + 1)
                # gathered_auc = [None] * (args.num_gpus + 1)
                gathered_auc = [None] * (all_proc + 1)
                torch.distributed.gather_object(float(0), gathered_ap, dst=args.num_gpus)
                torch.distributed.gather_object(float(0), gathered_auc, dst=args.num_gpus)
                # ap_tot += gathered_ap[:-1]
                ap_tot += gathered_ap[:1] + gathered_ap[2:]
                # auc_tot += gathered_auc[:-1]
                auc_tot += gathered_auc[:1] + gathered_auc[2:]
                multi_mfgs = list()
                multi_root = list()
                multi_ts = list()
                multi_eid = list()
                multi_block = list()
            pbar.update(1)
        ap = float(torch.tensor(ap_tot).mean())
        auc = float(torch.tensor(auc_tot).mean())
        return ap, auc


    best_ap = 0
    best_e = 0
    tap = 0
    tauc = 0
    # 开始epoch轮训练
    for e in range(train_param['epoch']):
        t_tot_s2 = time.time()
        mark = 0
        geshu = 0
        print('Epoch {:d}:'.format(e))
        time_sample = 0
        time_tot = 0
        time_train = 0
        if sampler is not None:
            sampler.reset()
        if mailbox is not None:
            mailbox.reset()
        # 定义每个batch实际训练数目为num_gpus * num_gpus * orig_batch_size
        train_param['batch_size'] = orig_batch_size
        # itr_tot表示训练阶段总共需要的轮次，然后每all_proc轮，进行一次训练
        # itr_tot = train_edge_end // train_param['batch_size'] // args.num_gpus * args.num_gpus
        itr_tot = train_edge_end // train_param['batch_size'] // all_proc * all_proc
        train_param['batch_size'] = math.ceil(train_edge_end / itr_tot)
        multi_mfgs = list()
        multi_root = list()
        multi_ts = list()
        multi_eid = list()
        multi_block = list()
        group_indexes = list()
        # index根据batch_size来进行分组
        group_indexes.append(np.array(df[:train_edge_end].index // train_param['batch_size']))
        if 'reorder' in train_param:
            # random chunk shceduling
            reorder = train_param['reorder']
            group_idx = list()
            for i in range(reorder):
                group_idx += list(range(0 - i, reorder - i))
            group_idx = np.repeat(np.array(group_idx), train_param['batch_size'] // reorder)
            group_idx = np.tile(group_idx, train_edge_end // train_param['batch_size'] + 1)[:train_edge_end]
            group_indexes.append(group_indexes[0] + group_idx)
            base_idx = group_indexes[0]
            for i in range(1, train_param['reorder']):
                additional_idx = np.zeros(train_param['batch_size'] // train_param['reorder'] * i) - 1
                group_indexes.append(np.concatenate([additional_idx, base_idx])[:base_idx.shape[0]])
        # with tqdm(total=itr_tot + max((val_edge_end - train_edge_end) // train_param['batch_size'] // args.num_gpus, 1) * args.num_gpus) as pbar:
        with tqdm(total=itr_tot + max((val_edge_end - train_edge_end) // train_param['batch_size'] // all_proc, 1) * all_proc) as pbar:
            for _, rows in df[:train_edge_end].groupby(group_indexes[random.randint(0, len(group_indexes) - 1)]):
                t_tot_s = time.time()
                # 获取每组的数据，定义好对应的root_nodes和ts，然后采样
                root_nodes = np.concatenate(
                    [rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
                ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
                if sampler is not None:
                    if 'no_neg' in sample_param and sample_param['no_neg']:
                        pos_root_end = root_nodes.shape[0] * 2 // 3
                        sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                    else:
                        sampler.sample(root_nodes, ts)
                    # 获取采样结果ret
                    ret = sampler.get_ret()
                    time_sample += ret[0].sample_time()
                # 根据采样得到的结果ret，生成对应的消息流图MFG
                if gnn_param['arch'] != 'identity':
                    mfgs = to_dgl_blocks(ret, sample_param['history'], cuda=False)
                else:
                    mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=False)
                multi_mfgs.append(mfgs)
                multi_root.append(root_nodes)
                multi_ts.append(ts)
                multi_eid.append(rows['Unnamed: 0'].values)
                if mailbox is not None and memory_param['deliver_to'] == 'neighbors':
                    multi_block.append(to_dgl_blocks(ret, sample_param['history'], reverse=True, cuda=False)[0][0])
                # if len(multi_mfgs) == args.num_gpus:
                if len(multi_mfgs) == all_proc:
                    # 设置模型状态为0，让gpu进程开始训练
                    # print("主进程设置模型状态为0，开始训练！")
                    # model_state = [0] * (args.num_gpus + 1)
                    model_state = [0] * (all_proc + 1)
                    my_model_state = [None]
                    torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
                    # multi_mfgs.append(None)
                    multi_mfgs.insert(1, None)
                    my_mfgs = [None]
                    # 将mfgs分发给其他gpu进程
                    # print("已经将mfgs分发给了其他gpu进程")
                    # print(multi_mfgs[0])
                    geshu += 1
                    torch.distributed.scatter_object_list(my_mfgs, multi_mfgs, src=args.num_gpus)
                    if mailbox is not None:
                        # multi_root.append(None)
                        multi_root.insert(1, None)
                        # multi_ts.append(None)
                        multi_ts.insert(1, None)
                        # multi_eid.append(None)
                        multi_eid.insert(1, None)
                        my_root = [None]
                        my_ts = [None]
                        my_eid = [None]
                        torch.distributed.scatter_object_list(my_root, multi_root, src=args.num_gpus)
                        torch.distributed.scatter_object_list(my_ts, multi_ts, src=args.num_gpus)
                        torch.distributed.scatter_object_list(my_eid, multi_eid, src=args.num_gpus)
                        if memory_param['deliver_to'] == 'neighbors':
                            # multi_block.append(None)
                            multi_block.insert(1, None)
                            my_block = [None]
                            torch.distributed.scatter_object_list(my_block, multi_block, src=args.num_gpus)
                    # print(multi_mfgs[0])
                    multi_mfgs = list()
                    multi_root = list()
                    multi_ts = list()
                    multi_eid = list()
                    multi_block = list()
                pbar.update(1)
            time_tot += time.time() - t_tot_s2
            print('Total time:', time_tot)
            print('Training time:', time_train)
            # 设置模型状态为5，开始收集损失值
            # model_state = [5] * (args.num_gpus + 1)
            model_state = [5] * (all_proc + 1)
            my_model_state = [None]
            torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
            # gathered_loss = [None] * (args.num_gpus + 1)
            gathered_loss = [None] * (all_proc + 1)
            torch.distributed.gather_object(float(0), gathered_loss, dst=args.num_gpus)
            # 计算损失值
            total_loss = np.sum(np.array(gathered_loss) * train_param['batch_size'])
            # 获取交叉验证集的验证结果
            ap, auc = eval('val')
            if ap > best_ap:
                best_e = e
                best_ap = ap
                # 如果发现了更好的，那么让gpu0将模型下载到本地，其他gpu不工作
                # model_state = [4] * (args.num_gpus + 1)
                model_state = [4] * (all_proc + 1)
                model_state[0] = 2
                my_model_state = [None]
                torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
                # for memory based models, testing after validation is faster
                # 获取交叉验证集更好结果状态下的测试集结果
                tap, tauc = eval('test')
        # wql add here(计算所有时间):
        print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss, ap, auc))
        print('\ttotal time:{:.2f}s sample time:{:.2f}s train time:{:.2f}s'.format(time_tot, time_sample, time_train))
        all_total_time += time_tot
        all_sample_time += time_sample
        all_train_time += time_train
        if (e + 1) % 10 == 0:
            print("in [{:d},{:d}], the mean total time cost of epoch: {:.4f}".format(e - 8, e + 1, all_total_time / 10))
            print("in [{:d},{:d}], the mean sample time cost of epoch: {:.4f}".format(e - 8, e + 1, all_sample_time / 10))
            print("in [{:d},{:d}], the mean train time cost of epoch: {:.4f}".format(e - 8, e + 1, all_train_time / 10))
            lis_total_time.append(all_total_time / 10)
            lis_sample_time.append(all_sample_time / 10)
            lis_train_time.append(all_train_time / 10)
            all_total_time = 0
            all_sample_time = 0
            all_train_time = 0
        # print("分发了: ", geshu, " 次！")


    print('Best model at epoch {}.'.format(best_e))
    print('\ttest ap:{:4f}  test auc:{:4f}'.format(tap, tauc))

    # 修改模型状态为-1，让所有进程结束
    # model_state = [-1] * (args.num_gpus + 1)
    model_state = [-1] * (all_proc + 1)
    my_model_state = [None]
    torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)


    for tmp in lis_total_time:
        print(tmp)
    print("-----------------")
    for tmp in lis_sample_time:
        print(tmp)
    print("-----------------")
    for tmp in lis_train_time:
        print(tmp)