import argparse
import logging
import os
import time


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--seed', type=int, default=0, help='random seed to use')
parser.add_argument('--num_gpus', type=int, default=4, help='number of gpus to use')
parser.add_argument('--omp_num_threads', type=int, default=8)
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--global_rank", type=int, default=-1)
parser.add_argument("--num_procs", type=int, default=-1)
args = parser.parse_args()

'''
python -m torch.distributed.launch --nproc_per_node=2 train_dist1.py --data WIKI_0 --config config/TGN.yml --num_gpus 1

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="10.214.151.192" --master_port=34567 \
train_dist2.py --data WIKI --config config/TGN.yml --num_gpus=1 --global_rank=0 --num_procs=2
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="10.214.151.192" --master_port=34567 \
train_dist2.py --data WIKI --config config/TGN.yml --num_gpus=1 --global_rank=1 --num_procs=2

rsync -avz /home/qcsun/wql_tgl/tgl-main qcsun@node192:/home/qcsun/wql_tgl

改版自->双机TGL原版(6.8.14)
已经实现了如下内容版本：
所有机子有一个cpu进程和一个gpu进程，cpu负责采样
直接把模型套到ddp上，就可以分布式采样了，不需要一个主cpu进程统筹管理模型调度
第一台机子负责处理前50%数据，第二台机子负责后50%数据
'''

# set which GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.local_rank)
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



set_seed(args.seed)
torch.distributed.init_process_group(backend='gloo', timeout=datetime.timedelta(0, 3600000), init_method='env://')
# ranks = [0, 1]
# nccl_group = torch.distributed.new_group(ranks=ranks, ll
# backend='nccl')

_node_feats, _edge_feats = load_feat(args.data)
dim_feats = [0, 0, 0, 0, 0, 0]
# print("准备开始读取节点特征和边特征")
if _node_feats is not None:
    # 创建内存共享节点/边特征变量node_feats和edge_feats
    # dim_feats[0]表示节点个数，dim_feats[1]表示节点dim，dim_feats[2]表示节点类型
    dim_feats[0] = _node_feats.shape[0]
    dim_feats[1] = _node_feats.shape[1]
    dim_feats[2] = _node_feats.dtype
    node_feats = create_shared_mem_array('node_feats', _node_feats.shape, dtype=_node_feats.dtype)
    node_feats.copy_(_node_feats)
    # print("正在读取节点特征，节点特征总共有", _node_feats.size())
    del _node_feats
else:
    node_feats = None
if _edge_feats is not None:
    dim_feats[3] = _edge_feats.shape[0]
    dim_feats[4] = _edge_feats.shape[1]
    dim_feats[5] = _edge_feats.dtype
    edge_feats = create_shared_mem_array('edge_feats', _edge_feats.shape, dtype=_edge_feats.dtype)
    edge_feats.copy_(_edge_feats)
    # print("正在读取边特征，边特征总共有", _edge_feats.size())
    del _edge_feats
else:
    edge_feats = None
# 进程第一次同步，保证edge_feats和node_feats被移动到内存中
torch.distributed.barrier()

sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
orig_batch_size = train_param['batch_size']
# 定义模型存储路径
if not os.path.isdir('models'):
    os.mkdir('models')
path_saver = ['models/{}_{}.pkl'.format(args.data, time.time())]
path_saver = path_saver[0]

# 获取图信息g和所有边信息df
g, df = load_graph(args.data)
num_nodes = [g['indptr'].shape[0] - 1]

# 等待cpu读取完g和df，然后将节点个数广播给所有进程
torch.distributed.barrier()
num_nodes = num_nodes[0]

mailbox = None
pinned_nfeat_buffs, pinned_efeat_buffs = None, None
# 如果需要memory，则创建mailbox的内存共享环境，mailbox可以理解为就是用来存memory的数据结构
if memory_param['type'] != 'none':
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
    mailbox = MailBox(memory_param, num_nodes, dim_feats[4], node_memory, node_memory_ts, mails, mail_ts, next_mail_pos,
                      update_mail_pos)


# 数据管道线程————暂时不知道该类作用
class DataPipelineThread(threading.Thread):

    def __init__(self, my_mfgs, my_root, my_ts, my_eid, my_block, stream):
        super(DataPipelineThread, self).__init__()
        # print("now mfgs in DPT: ", my_mfgs)
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
            prepare_input(mfgs, node_feats, edge_feats, pinned=True, nfeat_buffs=pinned_nfeat_buffs,
                          efeat_buffs=pinned_efeat_buffs, nids=nids, eids=eids)
            if mailbox is not None:
                # 一开始创建该类，将mailbox中保存的h传入mfgs[0]中
                mailbox.prep_input_mails(mfgs[0], use_pinned_buffers=True)
                self.mfgs = mfgs
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


# 如果是gpu操作
# 创建模型
model = GeneralModel(dim_feats[1], dim_feats[4], sample_param, memory_param, gnn_param, train_param).cuda()
find_unused_parameters = True if sample_param['history'] > 1 else False
# 将其设置为分布式模型
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
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


time_tmp1, time_tmp2, time_tmp3, time_tmp4, time_tmp5 = 0, 0, 0, 0, 0
def GPUWork(model_state, my_mfgs, my_root_nodes, my_ts, my_eid, my_block):
    global time_tmp1, time_tmp2, time_tmp3, time_tmp4, time_tmp5
    # 如果状态为2，则将模型信息下载到本地
    if model_state == 2:
        torch.save(model.state_dict(), path_saver)
    # 如果状态为3，则将从本地下载模型信息
    elif model_state == 3:
        model.load_state_dict(torch.load(path_saver, map_location=torch.device('cuda:0')))
    # 如果状态为0，则gpu开始训练
    elif model_state == 0:
        time_tmp = time.time()
        tot_loss = 0
        stream = torch.cuda.Stream()
        # 创建当前线程，用来读取新一轮的数据
        curr_thread = DataPipelineThread(my_mfgs, my_root_nodes, my_ts, my_eid, my_block, stream)
        curr_thread.start()
        # 让之前线程优先执行，等待之前线程执行完毕
        curr_thread.join()
        mfgs = curr_thread.get_mfgs()
        time_tmp1 += time.time() - time_tmp
        time_tmp = time.time()
        model.train()
        optimizer.zero_grad()
        pred_pos, pred_neg = model(mfgs)
        loss = creterion(pred_pos, torch.ones_like(pred_pos))
        loss += creterion(pred_neg, torch.zeros_like(pred_neg))
        time_tmp2 += time.time() - time_tmp
        time_tmp = time.time()
        loss.backward()
        optimizer.step()
        time_tmp3 += time.time() - time_tmp
        time_tmp = time.time()
        with torch.no_grad():
            tot_loss = float(loss)
        time_tmp4 += time.time() - time_tmp
        time_tmp = time.time()
        # 训练完之后更新一下mailbox和memory
        if mailbox is not None:
            with torch.no_grad():
                eid = curr_thread.get_eid()
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                root_nodes = curr_thread.get_root()
                ts = curr_thread.get_ts()
                block = curr_thread.get_block()
                # 更新初始根节点的memory和mailbox
                mailbox.update_mailbox(model.module.memory_updater.last_updated_nid,
                                       model.module.memory_updater.last_updated_memory, root_nodes, ts,
                                       mem_edge_feats, block)
                mailbox.update_memory(model.module.memory_updater.last_updated_nid,
                                      model.module.memory_updater.last_updated_memory,
                                      root_nodes,
                                      model.module.memory_updater.last_updated_ts)
                if memory_param['deliver_to'] == 'neighbors':
                    torch.distributed.barrier()
                    # ？这里的local_rank要不要改为global_rank
                    if args.local_rank == 0:
                        mailbox.update_next_mail_pos()
        time_tmp5 += time.time() - time_tmp
        return tot_loss, 0
    # 如果状态为1，则进行交叉验证
    elif model_state == 1:
        mfgs = mfgs_to_cuda(my_mfgs[0])
        prepare_input(mfgs, node_feats, edge_feats, pinned=True, nfeat_buffs=pinned_nfeat_buffs,
                      efeat_buffs=pinned_efeat_buffs)
        model.eval()
        with torch.no_grad():
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            pred_pos, pred_neg = model(mfgs)
            if mailbox is not None:
                eid = my_eid[0]
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                root_nodes = my_root_nodes[0]
                ts = my_ts[0]
                block = None
                if memory_param['deliver_to'] == 'neighbors':
                    block = my_block[0]
                mailbox.update_mailbox(model.module.memory_updater.last_updated_nid,
                                       model.module.memory_updater.last_updated_memory, root_nodes, ts,
                                       mem_edge_feats, block)
                mailbox.update_memory(model.module.memory_updater.last_updated_nid,
                                      model.module.memory_updater.last_updated_memory,
                                      root_nodes,
                                      model.module.memory_updater.last_updated_ts)
                if memory_param['deliver_to'] == 'neighbors':
                    torch.distributed.barrier()
                    if args.local_rank == 0:
                        mailbox.update_next_mail_pos()
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            ap = average_precision_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            return ap, auc
    return 0, 0


# 获取训练集和交叉验证集的终止位置
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]
test_edge_end = len(df)
test_edge_start = val_edge_end + args.global_rank * ((test_edge_end - val_edge_end) // args.num_procs)
if args.global_rank < args.num_procs - 1:
    test_edge_end = val_edge_end + (args.global_rank + 1) * ((test_edge_end - val_edge_end) // args.num_procs)
val_edge_start = train_edge_end + args.global_rank * ((val_edge_end - train_edge_end) // args.num_procs)
if args.global_rank < args.num_procs - 1:
    val_edge_end = train_edge_end + (args.global_rank + 1) * ((val_edge_end - train_edge_end) // args.num_procs)
train_edge_start = args.global_rank * (train_edge_end // args.num_procs)
if args.global_rank < args.num_procs - 1:
    train_edge_end = (args.global_rank + 1) * (train_edge_end // args.num_procs)
print(train_edge_start, " -> ", train_edge_end)
print(val_edge_start, " -> ", val_edge_end)
print(test_edge_start, " -> ", test_edge_end)

sampler = None
# 定义正负采样对象
if not ('no_sample' in sample_param and sample_param['no_sample']):
    # print(g['ts'])
    # print(type(g['ts']))
    # print(type(g['ts'][0]))
    sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                              sample_param['strategy'] == 'recent', sample_param['prop_time'],
                              sample_param['history'], float(sample_param['duration']))
    # print("wql add!!!!!!!!!!")
    # print(type(g['indptr']), len(g['indptr']))
    # print(type(g['indices']))
    # print(type(g['eid']))
    # print(type(g['ts']))
    # print(type(sample_param['num_thread']))
    time.sleep(10)
neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)


# 进行交叉验证和测试工作
def eval(mode='val'):
    itr_tol = 0
    if mode == 'val':
        eval_df = df[val_edge_start:val_edge_end]
        itr_tol = (val_edge_end - val_edge_start) // train_param['batch_size']
    elif mode == 'test':
        eval_df = df[test_edge_start:test_edge_end]
        itr_tol = (test_edge_end - test_edge_start) // train_param['batch_size']
    # 主要实现步骤与该函数后面的代码类似，这里不细加注释
    ap_tot = list()
    auc_tot = list()
    train_param['batch_size'] = orig_batch_size
    geshu = 0
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
        my_root = None
        my_ts = None
        my_eid = None
        my_block = None
        if mailbox is not None:
            my_root = root_nodes
            my_ts = ts
            my_eid = rows['Unnamed: 0'].values
            if memory_param['deliver_to'] == 'neighbors':
                my_block = to_dgl_blocks(ret, sample_param['history'], reverse=True, cuda=False)[0][0]

        ap, auc = GPUWork(1, [mfgs], [my_root], [my_ts], [my_eid], [my_block])
        gathered_ap = [None] * args.num_procs
        gathered_auc = [None] * args.num_procs
        if args.global_rank != 0:
            torch.distributed.gather_object(float(ap), None, dst=0)
            torch.distributed.gather_object(float(auc), None, dst=0)
        else:
            torch.distributed.gather_object(float(ap), gathered_ap, dst=0)
            torch.distributed.gather_object(float(auc), gathered_auc, dst=0)
            ap_tot += gathered_ap[:]
            auc_tot += gathered_auc[:]
        pbar.update(1)
        geshu += 1
        if geshu >= itr_tol:
            break
    if args.global_rank != 0:
        return 0, 0
    else:
        ap = float(torch.tensor(ap_tot).mean())
        auc = float(torch.tensor(auc_tot).mean())
        return ap, auc


best_ap = 0
best_e = 0
tap = 0
tauc = 0
time_sample_list, time_tot_list, time_train_list = list(), list(), list()
time_sample_all, time_tot_all, time_train_all = 0, 0, 0
# 开始epoch轮训练
for e in range(train_param['epoch']):
    mark = 0
    geshu = 0
    print('Epoch {:d}:'.format(e))
    time_sample, time_tot, time_train = 0, 0, 0
    if sampler is not None:
        sampler.reset()
    if mailbox is not None:
        mailbox.reset()
    # 定义每个batch实际训练数目为num_gpus * num_gpus * orig_batch_size
    train_param['batch_size'] = orig_batch_size
    # itr_tot表示训练阶段总共需要的轮次
    itr_tot = (train_edge_end - train_edge_start) // train_param['batch_size']
    group_indexes = list()
    # index根据batch_size来进行分组
    group_indexes.append(np.array(df[train_edge_start:train_edge_end].index // train_param['batch_size']))
    total_loss = 0
    geshu = 0
    with tqdm(total=itr_tot + max((val_edge_end - val_edge_start) // train_param['batch_size'], 1)) as pbar:
        for _, rows in df[train_edge_start:train_edge_end].groupby(group_indexes[0]):
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
            my_root = None
            my_ts = None
            my_eid = None
            my_block = None
            if mailbox is not None:
                my_root = root_nodes
                my_ts = ts
                my_eid = rows['Unnamed: 0'].values
                if memory_param['deliver_to'] == 'neighbors':
                    my_block = to_dgl_blocks(ret, sample_param['history'], reverse=True, cuda=False)[0][0]
            # print(mfgs)
            t_train_s = time.time()
            tmp_loss, a = GPUWork(0, [mfgs], [my_root], [my_ts], [my_eid], [my_block])
            time_train += time.time() - t_train_s
            total_loss += tmp_loss
            pbar.update(1)
            time_tot += time.time() - t_tot_s
            geshu += 1
            if geshu == itr_tot:
                break
        gathered_loss = [None] * args.num_procs
        if args.global_rank != 0:
            torch.distributed.gather_object(float(total_loss), None, dst=0)
        else:
            torch.distributed.gather_object(float(total_loss), gathered_loss, dst=0)
            total_loss = np.sum(np.array(gathered_loss) * train_param['batch_size'])
            print("the total loss is ", total_loss)
        # 获取交叉验证集的验证结果
        ap, auc = eval('val')
        # 获取测试集的验证结果
        if_eval_test = 0
        if args.global_rank == 0 and ap > best_ap:
            if_eval_test = 1
            best_e = e
            best_ap = ap
        my_eval_test = [None]
        eval_test = [if_eval_test] * args.num_procs
        if args.global_rank != 0:
            torch.distributed.scatter_object_list(my_eval_test, None, src=0)
        else:
            torch.distributed.scatter_object_list(my_eval_test, eval_test, src=0)
        if my_eval_test[0] == 1:
            tap, tauc = eval('test')
    if args.global_rank == 0:
        print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss, ap, auc))
        print('\ttotal time:{:.2f}s sample time:{:.2f}s train time:{:.2f}s'.format(time_tot, time_sample, time_train))
        time_sample_all += time_sample
        time_train_all += time_train
        time_tot_all += time_tot
        # print("now: {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(time_tmp1, time_tmp2, time_tmp3, time_tmp4, time_tmp5))
        time_tmp1, time_tmp2, time_tmp3, time_tmp4, time_tmp5 = 0, 0, 0, 0, 0
    if (e + 1) % 10 == 0:
        time_sample_list.append(time_sample_all / 10)
        time_train_list.append(time_train_all / 10)
        time_tot_list.append(time_tot_all / 10)
        print('every 10 epochs the mean total time:{:.2f}s sample time:{:.2f}s train time:{:.2f}s'
              .format(time_tot_all / 10, time_sample_all / 10, time_train_all / 10))
        time_sample_all, time_tot_all, time_train_all = 0, 0, 0

if args.global_rank == 0:
    print('Best model at epoch {}.'.format(best_e))
    print('\ttest ap:{:4f}  test auc:{:4f}'.format(tap, tauc))
    print("================")
    for i in time_tot_list:
        print("{:.2f}".format(i))
    print("================")
    for i in time_sample_list:
        print("{:.2f}".format(i))
    print("================")
    for i in time_train_list:
        print("{:.2f}".format(i))
    print("================")