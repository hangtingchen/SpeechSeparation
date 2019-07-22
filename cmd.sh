# you can change cmd.sh depending on what type of queue you are using.
# If you have no queueing system and want to run on a local machine, you
# can change all instances 'queue.pl' to run.pl (but be careful and run
# commands one by one: most recipes will exhaust the memory on your
# machine).  queue.pl works with GridEngine (qsub).  slurm.pl works
# with slurm.  Different queues are configured differently, with different
# queue names and different ways of specifying things like memory;
# to account for these differences you can create and edit the file
# conf/queue.conf to match your queue's configuration.  Search for
# conf/queue.conf in http://kaldi-asr.org/doc/queue.html for more information,
# or search for the string 'default_config' in utils/queue.pl or utils/slurm.pl.

#export train_cmd="queue.pl --mem 2G"
#export decode_cmd="queue.pl --mem 4G"

export train_cmd="queue.pl -q all.q -l arch=*64 --mem 4G --max-jobs-run 500"
export decode_cmd="queue.pl -q all.q -l arch=*64 --mem 2G --max-jobs-run 500 " #  --mem 1G "
export mkgraph_cmd="queue.pl --max-jobs-run 500 " # --mem 1G "
export cuda_cmd="queue.pl --gpu 1  --max-jobs-run 500 " #--mem 1G "
export cuda_cmd_run_pl="run.pl --gpu 1"
