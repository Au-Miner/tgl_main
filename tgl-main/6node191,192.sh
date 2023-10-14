#!/bin/bash
# running in 4 nodes: 191,192,197,198


# 检查端口是否被占用
PORT=34567
PID=$(lsof -t -i:$PORT)
# 如果端口被占用
if [[ ! -z $PID ]]; then
    echo "Port $PORT is in use by PID $PID. Terminating..."
    kill -9 $PID
    if [ $? -eq 0 ]; then
        echo "Successfully terminated process $PID."
    else
        echo "Failed to terminate process $PID. Exiting..."
        exit 1
    fi
else
    echo "Port $PORT is not in use."
fi



# 杀死正在使用 NVIDIA 设备文件的所有进程
echo "清除无关进程 释放端口"
fuser -v /dev/nvidia* | awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh



echo "准备清理以前内存"
rm -rf /dev/shm/mails /dev/shm/mail_ts /dev/shm/next_mail_pos /dev/shm/node_memory
rm -rf /dev/shm/node_memory_ts /dev/shm/edge_feats /dev/shm/update_mail_pos
#sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
echo "查看当前内存"
free -m

ip_addr=$(hostname -I | awk '{print $1}')
echo "抓取到的当前地址: $ip_addr"


TARGET_IPS=("10.214.151.192")

activate_and_run() {
    source $1/etc/profile.d/conda.sh
    conda activate $2
    echo "准备启动项目"
    python -m torch.distributed.launch --nproc_per_node=$3 --nnodes=2 --node_rank=$4 --master_addr="10.214.151.191" \
    --master_port=34567 train_node.py --data $5 \
    --config /home/qcsun/wql_tgl/tgl-main/config/$6.yml \
    --model /home/qcsun/wql_tgl/tgl-main/models/$5_$6.pkl \

    --config config/$6.yml --num_gpus=1
}

case "$ip_addr" in
    "10.214.151.191")
        echo "ip为node191"
        echo "准备分发文件"
        #rsync -avz /home/qcsun/wql2_dtgl/dtgl_main/ qcsun@node191:/home/qcsun/wql2_dtgl/dtgl_main/

        for ip in "${TARGET_IPS[@]}"; do
            rsync -avz --force /home/qcsun/wql_tgl/tgl-main qcsun@$ip:/home/qcsun/wql_tgl
        done

        export NCCL_SOCKET_IFNAME=em1,^br-2cd32c74f1f1
        activate_and_run "/home/qcsun/anaconda3" "tgl" 1 0 $1 $2
        ;;
    *)
        echo "ip为node192"
        export NCCL_SOCKET_IFNAME=em2,^em1,^br-6ca3f947e6e4
        activate_and_run "/home/qcsun/anaconda3" "tgl" 1 1 $1 $2
        ;;
esac
