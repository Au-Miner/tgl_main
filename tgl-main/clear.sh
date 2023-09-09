#!/bin/bash


echo "准备清理以前进程"
sid_str=$(ps -ef | grep ./run.sh)
sid=$(echo "$sid_str" | awk '{print $2}')
kill -9 $sid