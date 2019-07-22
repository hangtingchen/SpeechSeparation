#!/bin/bash

# author : chenhangting
# date : 2018/12/10

# The script is to train one speaker's neural network for ss1 system within the specific session.
. ./path_tf.sh
set -ue
# settings
nj=96
stage=1
test_set_percent=0.03
python_tf=lns/python3
pesq=~/usr/pesq/P862/Software/pesq
net=DNN
approach=IM
isSS2=false

echo "$0 $@"  # Print the command line for logging
. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh


if [ $# -ne 2 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 <audio-dir> <exp-dir>"
  echo -e >&2 "eg:\n  $0 data/CHiME5_mixed_audio/S03/P09 exp/ss1_training/S03/P09"
  exit 1
fi

orig_audio_dir=$1/orig
mixed_audio_dir=$1/mixed
feat_dir=$1/feat
dir=$2

mkdir -p $dir
if [ ! -d $orig_audio_dir ] || [ ! -d $mixed_audio_dir ];then
    echo >&2 "$0: Error: unable to find $orig_audio_dir or $mixed_audio_dir"
    exit 1
fi

# prepare training and testing set
if [ $stage -le 1 ];then
    echo "============$0 Stage 1============"
    find $orig_audio_dir/*.wav | perl -ne '$p=$_;chomp $_;@F=split "_";print "$F[$#F-1] $p";' > $dir/orig_audio_list.txt
    find $mixed_audio_dir/*.wav | perl -ne '$p=$_;chomp $_;@F=split "_";print "$F[$#F-1] $p";' > $dir/mixed_audio_list.txt
    join $dir/orig_audio_list.txt $dir/mixed_audio_list.txt > $dir/audio_list.txt
    rm $dir/orig_audio_list.txt $dir/mixed_audio_list.txt
    audio_total_num=`cat $dir/audio_list.txt | wc -l`
    echo "Dataset size $audio_total_num"
    python3 steps/split_dataset.py $dir/audio_list.txt $test_set_percent $dir/audio_list.train.txt $dir/audio_list.test.txt
fi

if [ $stage -le 1 ] && $isSS2;then
    echo "============$0 Stage 1============"
    echo "For ss2, the ss1 and ss2 samples are combined"
    cp $dir/audio_list.train.txt $dir/audio_list.train.txt.bak
    cat `echo $dir/audio_list.train.txt | sed 's?ss2?ss1?g'` $dir/audio_list.train.txt.bak | sort | uniq > $dir/audio_list.train.txt
fi

# For tensorflow
if [ $stage -le 2 ];then
    echo "============$0 Stage 2============"
    mkdir -p $feat_dir/train && mkdir -p $feat_dir/test
    for ds in train test;do  
            echo "Extract LSP feautures for $ds"
            if [ -f $feat_dir/$ds/$ds.tfrecords ];then rm $feat_dir/$ds/$ds.tfrecords;fi
            $python_tf -u utils/feats/extract_lsp_tf.py $dir/audio_list.$ds.txt \
                        $feat_dir/$ds/$ds.tfrecords $feat_dir/$ds
    done
fi

# training the network
tf_model_dir=$dir/tf_${net}_${approach}
if [ $stage -le 3 ];then
    echo "============$0 Stage 3============"
    TF_CPP_MIN_LOG_LEVEL=2 \
    $python_tf -u utils/tf_regression/run_regression.py \
        --input_dim 257 \
        --output_dim 257 \
        --left_context 3 \
        --right_context 3 \
        --num_hidden_layers 3 \
        --cell_dim 2048 \
        --proj_dim -1 \
        --recur_dim -1 \
        --mapping_mode $approach \
        --net_type $net \
        --model_dir $tf_model_dir \
        --input_mean $feat_dir/train/mixed.mean.npy \
        --input_std $feat_dir/train/mixed.std.npy \
        --output_mean $feat_dir/train/speech.mean.npy \
        --output_std $feat_dir/train/speech.std.npy \
        train \
        --batch_size 64 \
        --epochs 10 \
        --lr 0.003 \
        --train_data $feat_dir/train/train.tfrecords \
        --valid_data $feat_dir/test/test.tfrecords \
        --check_times_one_epoch 2
fi

# infer the test set
if [ $stage -le 4 ];then
    echo "============$0 Stage 4============"
    TF_CPP_MIN_LOG_LEVEL=2 \
    $python_tf -u utils/tf_regression/run_regression.py \
            --input_dim 257 \
            --output_dim 257 \
            --left_context 3 \
            --right_context 3 \
            --num_hidden_layers 3 \
            --cell_dim 2048 \
            --proj_dim -1 \
            --recur_dim -1 \
            --mapping_mode $approach \
            --net_type $net \
            --model_dir $tf_model_dir \
            --input_mean $feat_dir/train/mixed.mean.npy \
            --input_std $feat_dir/train/mixed.std.npy \
            --output_mean $feat_dir/train/speech.mean.npy \
            --output_std $feat_dir/train/speech.std.npy \
            infer \
            --infer_data $feat_dir/test/test.tfrecords \
            --infer_out_dir $feat_dir/test/infer
fi

# calculate pesq
if [ $stage -le 5 ];then
    echo "============$0 Stage 5============"
    if [ -f $dir/model_pesq_list.txt ];then
        rm $dir/model_pesq_list.txt
    fi
    touch $dir/model_pesq_list.txt
    cat $dir/audio_list.test.txt | while read line;do
        org_audio=`echo $line | awk '{print $2}'`
        mixed_audio=`echo $line | awk '{print $3}'`
        base_mixed_audio=`basename $mixed_audio`
        infer_audio=$feat_dir/test/infer/${base_mixed_audio}
        mixed_audio_pesq=`$pesq +16000 ${org_audio} ${mixed_audio} | tail -1 | awk -F= '{print $2}'`
        infer_audio_pesq=`$pesq +16000 ${org_audio} ${infer_audio} | tail -1 | awk -F= '{print $2}'`
        # echo "${base_mixed_audio}${mixed_audio_pesq}${infer_audio_pesq}"
        echo "${base_mixed_audio}${mixed_audio_pesq}${infer_audio_pesq}" >> $dir/model_pesq_list.txt
    done
    python3 -u utils/tf_regression/pesq2report.py $dir/model_pesq_list.txt $dir/model_pesq_report.txt
fi

