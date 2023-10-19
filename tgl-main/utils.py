import torch
import os
import yaml
import dgl
import time
import pandas as pd
import numpy as np
from kvstore import *



def load_feat(d, rand_de=0, rand_dn=0):
    node_feats = None
    if os.path.exists('/home/qcsun/DATA/{}/node_features.pt'.format(d)):
        node_feats = torch.load('/home/qcsun/DATA/{}/node_features.pt'.format(d))
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists('/home/qcsun/DATA/{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('/home/qcsun/DATA/{}/edge_features.pt'.format(d))
        print("edge_feats.dtype111: ", edge_feats.dtype)
        print(edge_feats.shape)
        # if edge_feats.dtype == torch.bool:
        #     edge_feats = edge_feats.type(torch.float32)
        #     # edge_feats = edge_feats.to(torch.float32)
        #     edge_feats = edge_feats.numpy()
    if rand_de > 0:
        if d == 'LASTFM':
            edge_feats = torch.randn(1293103, rand_de)
        elif d == 'MOOC':
            edge_feats = torch.randn(411749, rand_de)
    if rand_dn > 0:
        if d == 'LASTFM':
            node_feats = torch.randn(1980, rand_dn)
        elif d == 'MOOC':
            node_feats = torch.randn(7144, rand_dn)
    return node_feats, edge_feats

def load_graph(d):
    df = pd.read_csv('/home/qcsun/DATA/{}/edges.csv'.format(d))
    g = np.load('/home/qcsun/DATA/{}/ext_full.npz'.format(d))
    return g, df

def parse_config(f):
    conf = yaml.safe_load(open(f, 'r'))
    sample_param = conf['sampling'][0]
    memory_param = conf['memory'][0]
    gnn_param = conf['gnn'][0]
    train_param = conf['train'][0]
    return sample_param, memory_param, gnn_param, train_param

def to_dgl_blocks(ret, hist, reverse=False, cuda=True):
    mfgs = list()
    for r in ret:
        if not reverse:
            b = dgl.create_block((r.col(), r.row()), num_src_nodes=r.dim_in(), num_dst_nodes=r.dim_out())
            b.srcdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_dst_nodes():]
            b.srcdata['ts'] = torch.from_numpy(r.ts())
        else:
            b = dgl.create_block((r.row(), r.col()), num_src_nodes=r.dim_out(), num_dst_nodes=r.dim_in())
            b.dstdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_src_nodes():]
            b.dstdata['ts'] = torch.from_numpy(r.ts())
        b.edata['ID'] = torch.from_numpy(r.eid())
        if cuda:
            mfgs.append(b.to('cuda:0'))
        else:
            mfgs.append(b)
    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    return mfgs

def node_to_dgl_blocks(root_nodes, ts, cuda=True):
    mfgs = list()
    b = dgl.create_block(([],[]), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0])
    b.srcdata['ID'] = torch.from_numpy(root_nodes)
    b.srcdata['ts'] = torch.from_numpy(ts)
    if cuda:
        mfgs.insert(0, [b.to('cuda:0')])
    else:
        mfgs.insert(0, [b])
    return mfgs

def mfgs_to_cuda(mfgs):
    for mfg in mfgs:
        for i in range(len(mfg)):
            mfg[i] = mfg[i].to('cuda:0')
    return mfgs



def prepare_input(mfgs, node_feats_not_None, edge_feats_not_None, combine_first=False, pinned=False, nfeat_buffs=None, efeat_buffs=None, nids=None, eids=None):
    if node_feats_not_None:
        for b in mfgs[0]:
            srch = pull_remote(b.srcdata['ID'].long(), 'node', get_rank()).float()
            # srch = node_feats[b.srcdata['ID'].long()].float()
            b.srcdata['h'] = srch.cuda()
    if edge_feats_not_None:
        for mfg in mfgs:
            for b in mfg:
                if b.num_src_nodes() > b.num_dst_nodes():
                    device = torch.device('cpu')
                    # print("sample到了", len(b.edata['ID'].cpu().long()))
                    srch = pull_remote(b.edata['ID'].long().to(device), 'edge', get_rank())
                    # srch = edge_feats[b.edata['ID'].long().to(device)]
                    srch = srch.type(torch.float32)
                    b.edata['f'] = srch.cuda()
    return mfgs

def get_ids(mfgs, node_feats_not_None, edge_feats_not_None):
    nids = list()
    eids = list()
    if node_feats_not_None:
        for b in mfgs[0]:
            nids.append(b.srcdata['ID'].long())
    if edge_feats_not_None:
        for mfg in mfgs:
            for b in mfg:
                if 'ID' in b.edata:
                    eids.append(b.edata['ID'].long())
    return nids, eids

def get_pinned_buffers(sample_param, batch_size, node_feats_not_None, node_feats_shape1, edge_feats_not_None, edge_feats_shape1):
    pinned_nfeat_buffs = list()
    pinned_efeat_buffs = list()
    limit = int(batch_size * 3.3)
    if 'neighbor' in sample_param:
        for i in sample_param['neighbor']:
            limit *= i + 1
            if edge_feats_not_None:
                for _ in range(sample_param['history']):
                    pinned_efeat_buffs.insert(0, torch.zeros((limit, node_feats_shape1), pin_memory=True))
    if node_feats_not_None:
        for _ in range(sample_param['history']):
            pinned_nfeat_buffs.insert(0, torch.zeros((limit, edge_feats_shape1), pin_memory=True))
    return pinned_nfeat_buffs, pinned_efeat_buffs
