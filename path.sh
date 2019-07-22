export KALDI_ROOT=/nobackup/f2/asr/chenhangting/Project/chime5/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
#export PATH=/nobackup/s1/asr/zhangyike/Anaconda3/bin/:$PATH
export PATH=./lns/:$PATH

#Matlab
export PATH=/nobackup/s1/asr/zhangyike/Matlab2015/bin:$PATH