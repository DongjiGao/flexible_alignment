#!/bin/bash

# 2022 Dongji Gao (Johns Hopkins University)

stage=0

# bss and vad setting
wav_file="data/raw/wav.scp"
tmp_dir="data/tmp"
bss_output_dir="data/bss_audio"
vad_output_dir="data/raw"

# lang setting
oov="<UNK>"
lang_dir="data/lang_ntu"
graph_dir="exp/graph"

# flexible_alignment setting
model="model/robust_seame_eng_ntu/checkpoint-54500"
dataset="dataset/ntu_raw_test_small"
text="../test/data/raw_toy/text"
output_dir=""

# others
log_dir="exp/log"
nj=1

. ./path.sh
. ./cmd.sh
. utils/parse_options.sh

set -e

if [ ${stage} -le 0 ]; then
  echo "$0: stage 0, doing BSS"
  if [ ${nj} -le 1 ]; then
    local/prepare_bss.py ${wav_file} ${tmp_dir} ${bss_output_dir}
  else
    split_wav=""
    for n in $(seq ${nj}); do
      split_wav="${split_wav} ${tmp_dir}/wav_${n}.scp"
    done
    #utils/split_scp.pl ${wav_file} ${split_wav}
    # TODO: split wav.scp based on whole session
    # TODO: write wav.scp
    ${alignment_cmd} JOB=1:${nj} ${log_dir}/bss_JOB.log local/prepare_bss.py \
      ${tmp_dir}/wav_JOB.scp \
      ${tmp_dir} \
      ${bss_output_dir}
  fi
fi

if [ ${stage} -le 1 ]; then
  echo "$0: stage 1, doing VAD on BSSed audio"
  ${alignment_cmd} ${log_dir}/vad.log local/vad.py \
    --ref-segment "data/ntu/segments" \
    --collar 0.2 \
    --gap 0.5 \
    --metric "precision_recall" \
    ${wav_file} \
    ${vad_output_dir}
fi

if [ ${stage} -le 2 ]; then
  echo "$0: stage 2, preparing lang for flexible alignment"
  local/prepare_data.sh \
    --oov ${oov} \
    ${lang} \
    ${graph_dir}
fi

if [ ${stage} -le 3 ]; then
  echo "$0: stage 3, doing flexible alignment"
  ${alignment_cmd} ${log_dir}/align.log align.py \
   ${model} \
   ${dataset} \
   ${text} \
   ${ses2spk} \
   ${lang} \
   ${graph_dir} \
   ${output_dir}
fi

exit 0;
