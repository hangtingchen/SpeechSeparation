export PATH=./lns/:$PATH
export CUDA_VISIBLE_DEVICES=5
#cuda9.0
export CUDA_HOME=/usr/local/cuda-9.0
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# cudnn7.1
#export LD_LIBRARY_PATH=/nobackup/s1/asr/zhangyike/cudnn7.1/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/nobackup/s1/asr/zhangyike/cudnn7.0/cuda/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=4
