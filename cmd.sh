# NTU Collaborative data

queue_conf="conf/clsp_config_gpus.conf"
a5000_conf="conf/a5000.conf"
export finetune_cmd="queue.pl -q p.q --config ${a5000_conf} --gpu 2 -V "
export alignment_cmd="queue.pl -q g.q --config ${queue_conf} --gpu 1 -V "
export alignment_cmd="queue.pl -l hostname=c* --gpu 1"
export alignment_cmd="queue.pl -q p.q --config ${a5000_conf} --gpu 1 -V "
export CUDA_VISIBLE_DEVICES=1
