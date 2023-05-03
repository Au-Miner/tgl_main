import argparse
import os

'''
在主进程中，通过 scatter_object_list 和 gather_object 等函数将数据、模型和评估结果分配给各个计算节点，并接收计算节点返回的结果并进行聚合。
在计算节点中，则通过 scatter_object_list 和 gather_object 等函数接收主进程发来的数据和模型，并对这些数据进行处理并返回评估结果。
整个分布式训练过程中，主进程和计算节点之间通过 MPI 协议进行通信，以完成数据和模型的传输和同步。
'''
parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--seed', type=int, default=0, help='random seed to use')
parser.add_argument('--num_gpus', type=int, default=4, help='number of gpus to use')
parser.add_argument('--model',  type=str, help='path to model file')
parser.add_argument('--batch_size',  type=int, default=4000, help='batch size to generate node embeddings')
parser.add_argument('--omp_num_threads', type=int, default=16)
parser.add_argument("--local_rank", type=int, default=-1)
args=parser.parse_args()

# set which GPU to use
# 使用指定的GPU组 在一机器多卡的机器中,我们可以指定使用某几台GPU,而剩下的GPU在程序中不会被使用
if args.local_rank < args.num_gpus:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.local_rank)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = str(args.omp_num_threads)
os.environ['MKL_NUM_THREADS'] = str(args.omp_num_threads)

import torch
import dgl
import random
import math
import hashlib
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
torch.distributed.init_process_group(backend='gloo')
nccl_group = torch.distributed.new_group(ranks=list(range(args.num_gpus)), backend='nccl')

if args.local_rank == 0:
    # 从'DATA/{}/node_features.pt'中加载点特征
    _node_feats, _edge_feats = load_feat(args.data)
dim_feats = [0, 0, 0, 0, 0, 0]
if args.local_rank == 0:
    if _node_feats is not None:
        dim_feats[0] = _node_feats.shape[0]
        dim_feats[1] = _node_feats.shape[1]
        dim_feats[2] = _node_feats.dtype
        node_feats = create_shared_mem_array('node_feats', _node_feats.shape, dtype=_node_feats.dtype)
        node_feats.copy_(_node_feats)
        del _node_feats
    else:
        node_feats = None
    if _edge_feats is not None:
        dim_feats[3] = _edge_feats.shape[0]
        dim_feats[4] = _edge_feats.shape[1]
        dim_feats[5] = _edge_feats.dtype
        edge_feats = create_shared_mem_array('edge_feats', _edge_feats.shape, dtype=_edge_feats.dtype)
        edge_feats.copy_(_edge_feats)
        del _edge_feats
    else: 
        edge_feats = None
torch.distributed.barrier()
torch.distributed.broadcast_object_list(dim_feats, src=0)
if args.local_rank > 0 and args.local_rank < args.num_gpus:
    node_feats = None
    edge_feats = None
    if os.path.exists('DATA/{}/node_features.pt'.format(args.data)):
        node_feats = get_shared_mem_array('node_feats', (dim_feats[0], dim_feats[1]), dtype=dim_feats[2])
    if os.path.exists('DATA/{}/edge_features.pt'.format(args.data)):
        edge_feats = get_shared_mem_array('edge_feats', (dim_feats[3], dim_feats[4]), dtype=dim_feats[5])
# 通过config地址来获取具体变量的参数
sample_param, memory_param, gnn_param, train_param = parse_config(args.config)

path_saver = args.model

if args.local_rank == args.num_gpus:
    # 加载全部图的信息，g为ext_full，df为edges
    g, df = load_graph(args.data)
    num_nodes = [g['indptr'].shape[0] - 1]
else:
    num_nodes = [None]
torch.distributed.barrier()
torch.distributed.broadcast_object_list(num_nodes, src=args.num_gpus)
num_nodes = num_nodes[0]

mailbox = None
if memory_param['type'] != 'none':
    if args.local_rank == 0:
        node_memory = create_shared_mem_array('node_memory', torch.Size([num_nodes, memory_param['dim_out']]), dtype=torch.float32)
        node_memory_ts = create_shared_mem_array('node_memory_ts', torch.Size([num_nodes]), dtype=torch.float32)
        mails = create_shared_mem_array('mails', torch.Size([num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_feats[4]]), dtype=torch.float32)
        mail_ts = create_shared_mem_array('mail_ts', torch.Size([num_nodes, memory_param['mailbox_size']]), dtype=torch.float32)
        next_mail_pos = create_shared_mem_array('next_mail_pos', torch.Size([num_nodes]), dtype=torch.long)
        update_mail_pos = create_shared_mem_array('update_mail_pos', torch.Size([num_nodes]), dtype=torch.int32)
        # 执行完缓存任务后，也阻塞掉
        # 当pytorch发现所有的进程都进入了barrier（），就会打开所有的barrier，所有的进程都可以继续进行
        torch.distributed.barrier()
        node_memory.zero_()
        node_memory_ts.zero_()
        mails.zero_()
        mail_ts.zero_()
        next_mail_pos.zero_()
        update_mail_pos.zero_()
    else:
        # 当前进程如果不是主进程，就让pytorch对它进行阻塞，也就是暂停运行
        torch.distributed.barrier()
        # 当主进程分享完信息后，其他进程获取分享的信息
        node_memory = get_shared_mem_array('node_memory', torch.Size([num_nodes, memory_param['dim_out']]), dtype=torch.float32)
        node_memory_ts = get_shared_mem_array('node_memory_ts', torch.Size([num_nodes]), dtype=torch.float32)
        mails = get_shared_mem_array('mails', torch.Size([num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_feats[4]]), dtype=torch.float32)
        mail_ts = get_shared_mem_array('mail_ts', torch.Size([num_nodes, memory_param['mailbox_size']]), dtype=torch.float32)
        next_mail_pos = get_shared_mem_array('next_mail_pos', torch.Size([num_nodes]), dtype=torch.long)
        update_mail_pos = get_shared_mem_array('update_mail_pos', torch.Size([num_nodes]), dtype=torch.int32)
    mailbox = MailBox(memory_param, num_nodes, dim_feats[4], node_memory, node_memory_ts, mails, mail_ts, next_mail_pos, update_mail_pos)

# 非主进程
if args.local_rank < args.num_gpus:
    # GPU worker process
    model = GeneralModel(dim_feats[1], dim_feats[4], sample_param, memory_param, gnn_param, train_param).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], process_group=nccl_group, output_device=args.local_rank)
    model.load_state_dict(torch.load(path_saver, map_location=torch.device('cuda:0')))
    creterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
    while True:
        my_model_state = [None]
        model_state = [None] * (args.num_gpus + 1)
        # 将主设备上的模型参数拆分成多个子列表，并分配到所有 GPU 设备上
        torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
        # 当 my_model_state[0] 为 -1 时，表示分布式训练已经结束；当其为 4 时，表示当前批次需要跳过； 当其为 2 时，表示主设备将模型参数保存下来以备以后使用；当其为 3 时，表示其他设备需要加载主设备保存的模型参数
        if my_model_state[0] == -1:
            break
        elif my_model_state[0] == 4:
            continue
        elif my_model_state[0] == 2:
            torch.save(model.state_dict(), path_saver)
            continue
        elif my_model_state[0] == 3:
            model.load_state_dict(torch.load(path_saver, map_location=torch.device('cuda:0')))
            continue
        my_mfgs = [None]
        multi_mfgs = [None] * (args.num_gpus + 1)
        # 使用 torch.distributed.scatter_object_list 函数将主设备上的输入数据拆分成多个子列表，并分配到所有 GPU 设备上。这些数据包括节点特征、边特征等信息
        torch.distributed.scatter_object_list(my_mfgs, multi_mfgs, src=args.num_gpus)
        mfgs = mfgs_to_cuda(my_mfgs[0])
        prepare_input(mfgs, node_feats, edge_feats)
        # 需要对模型进行训练
        if my_model_state[0] == 0:
            model.train()
            optimizer.zero_grad()
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            pred_pos, pred_neg = model(mfgs)
            loss = creterion(pred_pos, torch.ones_like(pred_pos))
            loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            loss.backward()
            optimizer.step()
            if mailbox is not None:
                with torch.no_grad():
                    my_root = [None]
                    multi_root = [None] * (args.num_gpus + 1)
                    my_ts = [None]
                    multi_ts = [None] * (args.num_gpus + 1)
                    my_eid = [None]
                    multi_eid = [None] * (args.num_gpus + 1)
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
                        multi_block = [None] * (args.num_gpus + 1)
                        torch.distributed.scatter_object_list(my_block, multi_block, src=args.num_gpus)
                        block = my_block[0]
                    mailbox.update_mailbox(model.module.memory_updater.last_updated_nid, model.module.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                    mailbox.update_memory(model.module.memory_updater.last_updated_nid, model.module.memory_updater.last_updated_memory, model.module.memory_updater.last_updated_ts)
                    if memory_param['deliver_to'] == 'neighbors':
                        torch.distributed.barrier(group=nccl_group)
                        if args.local_rank == 0:
                            mailbox.update_next_mail_pos()
            torch.distributed.gather_object(float(loss), None, dst=args.num_gpus)
        # 需要对模型进行评估
        elif my_model_state[0] == 1:
            model.eval()
            with torch.no_grad():
                if mailbox is not None:
                    mailbox.prep_input_mails(mfgs[0])
                pred_pos, pred_neg = model(mfgs)
                if mailbox is not None:
                    my_root = [None]
                    multi_root = [None] * (args.num_gpus + 1)
                    my_ts = [None]
                    multi_ts = [None] * (args.num_gpus + 1)
                    my_eid = [None]
                    multi_eid = [None] * (args.num_gpus + 1)
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
                        multi_block = [None] * (args.num_gpus + 1)
                        torch.distributed.scatter_object_list(my_block, multi_block, src=args.num_gpus)
                        block = my_block[0]
                    mailbox.update_mailbox(model.module.memory_updater.last_updated_nid, model.module.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                    mailbox.update_memory(model.module.memory_updater.last_updated_nid, model.module.memory_updater.last_updated_memory, model.module.memory_updater.last_updated_ts)
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
        # 需要获取模型的嵌入表示
        elif my_model_state[0] == 5:
            model.eval()
            with torch.no_grad():
                if mailbox is not None:
                    mailbox.prep_input_mails(mfgs[0])
                emb = model.module.get_emb(mfgs).detach().cpu()
                torch.distributed.gather_object(emb, None, dst=args.num_gpus)
else:
    # hosting process
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]
    sampler = None
    if not ('no_sample' in sample_param and sample_param['no_sample']):
        sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                  sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                                  sample_param['strategy']=='recent', sample_param['prop_time'],
                                  sample_param['history'], float(sample_param['duration']))
    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)

    ldf = pd.read_csv('DATA/{}/labels.csv'.format(args.data))
    args.batch_size = math.ceil(len(ldf) / (len(ldf) // args.batch_size // args.num_gpus * args.num_gpus))
    train_param['batch_size'] = math.ceil(len(df) / (len(df) // train_param['batch_size'] // args.num_gpus * args.num_gpus))

    processed_edge_id = 0
    def forward_model_to(time):
        global processed_edge_id
        if processed_edge_id >= len(df):
            return
        while df.time[processed_edge_id] < time:
            # print('curr:',processed_edge_id,df.time[processed_edge_id],'target:',time)
            multi_mfgs = list()
            multi_root = list()
            multi_ts = list()
            multi_eid = list()
            multi_block = list()
            for _ in range(args.num_gpus):
                if processed_edge_id >= len(df):
                    break
                # 从 df 中取出一定数量的数据，并将其转换为包含源节点、目的节点和时间戳的三元组
                rows = df[processed_edge_id:min(len(df), processed_edge_id + train_param['batch_size'])]
                # 根据源节点、目的节点和负样本采样器生成节点 ID，并将这些节点 ID 存储到 numpy 数组中
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
                ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
                # 如果存在采样器，使用采样器对节点 ID 进行采样，并获取采样结果
                if sampler is not None:
                    if 'no_neg' in sample_param and sample_param['no_neg']:
                        pos_root_end = root_nodes.shape[0] * 2 // 3
                        sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                    else:
                        sampler.sample(root_nodes, ts)
                    ret = sampler.get_ret()
                # 根据采样结果将节点 ID 和时间戳转换为 DGL 图块（MFG）
                if gnn_param['arch'] != 'identity':
                    mfgs = to_dgl_blocks(ret, sample_param['history'], cuda=False)
                else:
                    mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=False)
                # 将生成的 MFG、节点 ID、时间戳和边 ID 分别添加到多个列表中
                multi_mfgs.append(mfgs)
                multi_root.append(root_nodes)
                multi_ts.append(ts)
                multi_eid.append(rows['Unnamed: 0'].values)
                if mailbox is not None and memory_param['deliver_to'] == 'neighbors':
                    multi_block.append(to_dgl_blocks(ret, sample_param['history'], reverse=True, cuda=False)[0][0])
                # 将处理过的数据数量加上批次大小
                processed_edge_id += train_param['batch_size']
            if processed_edge_id >= len(df):
                return
            # 对于每个节点，该函数会将处理的 MFG、节点 ID、时间戳和边 ID 分别发送给相应的计算节点
            model_state = [1] * (args.num_gpus + 1)
            my_model_state = [None]
            torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
            multi_mfgs.append(None)
            my_mfgs = [None]
            torch.distributed.scatter_object_list(my_mfgs, multi_mfgs, src=args.num_gpus)
            if mailbox is not None:
                multi_root.append(None)
                multi_ts.append(None)
                multi_eid.append(None)
                my_root = [None]
                my_ts = [None]
                my_eid = [None]
                torch.distributed.scatter_object_list(my_root, multi_root, src=args.num_gpus)
                torch.distributed.scatter_object_list(my_ts, multi_ts, src=args.num_gpus)
                torch.distributed.scatter_object_list(my_eid, multi_eid, src=args.num_gpus)
                if memory_param['deliver_to'] == 'neighbors':
                    multi_block.append(None)
                    my_block = [None]
                    torch.distributed.scatter_object_list(my_block, multi_block, src=args.num_gpus)
            # 接收计算节点返回的模型状态和评估结果（AP 和 AUC），然后进行聚合，并更新全局模型状态和评估结果
            gathered_ap = [None] * (args.num_gpus + 1)
            gathered_auc = [None] * (args.num_gpus + 1)
            torch.distributed.gather_object(float(0), gathered_ap, dst=args.num_gpus)
            torch.distributed.gather_object(float(0), gathered_auc, dst=args.num_gpus)
            if processed_edge_id >= len(df):
                break

    embs = list()
    multi_mfgs = list()
    for _, rows in tqdm(ldf.groupby(ldf.index // args.batch_size)):
        root_nodes = rows.node.values.astype(np.int32)
        ts = rows.time.values.astype(np.float32)
        if args.data == 'MAG':
            # allow paper to sample neighbors
            ts += 1
        if sampler is not None:
            sampler.sample(root_nodes, ts)
            ret = sampler.get_ret()
        if gnn_param['arch'] != 'identity':
            mfgs = to_dgl_blocks(ret, sample_param['history'], cuda=False)
        else:
            mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=False)
        multi_mfgs.append(mfgs)
        if len(multi_mfgs) == args.num_gpus:
            forward_model_to(ts[-1])
            model_state = [5] * (args.num_gpus + 1)
            my_model_state = [None]
            torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
            multi_mfgs.append(None)
            my_mfgs = [None]
            torch.distributed.scatter_object_list(my_mfgs, multi_mfgs, src=args.num_gpus)
            multi_embs = [None] * (args.num_gpus + 1)
            torch.distributed.gather_object(None, multi_embs, dst=args.num_gpus)
            embs += multi_embs[:-1]
            multi_mfgs = list()
        
    emb_file_name = hashlib.md5(str(torch.load(args.model, map_location=torch.device('cpu'))).encode('utf-8')).hexdigest() + '.pt'
    if not os.path.isdir('embs'):
        os.mkdir('embs')
    embs = torch.cat(embs, dim=0)
    print('Embedding shape:', embs.shape)
    torch.save(embs, 'embs/' + emb_file_name)

    # let all process exit
    model_state = [-1] * (args.num_gpus + 1)
    my_model_state = [None]
    torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)