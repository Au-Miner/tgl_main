#!/bin/bash



echo "准备清理以前内存"
rm -rf /dev/shm/mails /dev/shm/mail_ts /dev/shm/next_mail_pos /dev/shm/node_memory
rm -rf /dev/shm/node_memory_ts /dev/shm/edge_feats /dev/shm/update_mail_pos
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
echo "查看当前内存"
free -m

echo "查看当前ip地址"
ip_addrs=$(ip addr)
target_line=$(echo "$ip_addrs" | grep 'inet 10.214.151.192')
ip_addr=$(echo "$target_line" | grep -oP 'inet \K[\d.]+')


if [ "$ip_addr" = "10.214.151.192" ]; then
    echo "ip为node192"
    echo "准备分发文件"
    rsync -avz /home/qcsun/wql_tgl/tgl-main qcsun@node191:/home/qcsun/wql_tgl
    echo "切换conda环境"
    source /home/qcsun/anaconda3/etc/profile.d/conda.sh
    conda activate tgl2
    echo "准备启动项目"
    python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="10.214.151.192" \
    --master_port=34567 train_dist_tmp.py --data $1 --config config/TGN.yml --num_gpus=1

else
    echo "ip为node191"
    echo "切换conda环境"
    source /home/qcsun/enter/etc/profile.d/conda.sh
    conda activate tgl
    echo "准备启动项目"
    python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="10.214.151.192" \
    --master_port=34567 train_dist_tmp.py --data $1 --config config/TGN.yml --num_gpus=1

fi



rsync -avz /home/qcsun/wql_tgl/tgl-main qcsun@node191:/home/qcsun/wql_tgl
rsync -avz qcsun@node192:/home/qcsun/DATA/REDDIT /home/qcsun/DATA


python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="10.214.151.192" \
--master_port=34567 train_dist9.7.24.py --data WIKI --config config/TGN.yml --num_gpus=1

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="10.214.151.192" \
--master_port=34567 train_dist9.7.24.py --data WIKI --config config/TGN.yml --num_gpus=1

###4

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=4 --node_rank=0 --master_addr="10.214.151.192" \
--master_port=34567 train_dist2.py --data REDDIT --config config/TGN.yml --num_gpus=1

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=1 --master_addr="10.214.151.192" \
--master_port=34567 train_dist2.py --data REDDIT --config config/TGN.yml --num_gpus=1

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=2 --master_addr="10.214.151.192" \
--master_port=34567 train_dist2.py --data REDDIT --config config/TGN.yml --num_gpus=1

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=3 --master_addr="10.214.151.192" \
--master_port=34567 train_dist2.py --data REDDIT --config config/TGN.yml --num_gpus=1




rsync -avz /home/qcsun/wql_tgl/tgl-main qcsun@node198:/home/qcsun/wql_tgl
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="10.214.151.191" \
--master_port=34567 train_dist2.py --data WIKI --config config/TGN.yml --num_gpus=1

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="10.214.151.191" \
--master_port=34567 train_dist2.py --data WIKI --config config/TGN.yml --num_gpus=1


python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="10.214.151.191" \
--master_port=34567 train_dist2.py --data WIKI --config config/TGN.yml --num_gpus=1