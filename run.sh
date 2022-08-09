#!/bin/bash

# 2022 Dongji Gao (Johns Hopkins University)

set -eu

stage=0

base=""
# single or multi
spk=""
suffix=""
use_xvector=""


# bss and vad setting
raw_wav_file=""
wav_file=""
tmp_dir=""
bss_output_dir=""
vad_output_dir=""

# lang setting
oov="<UNK>"
lang_dir="data/lang"
graph_dir="exp/graph/${base}_${spk}${suffix}"

# flexible_alignment setting
model=""
dataset=""
data_dir="data/${base}"
text="${data_dir}/text"
ses2spk="${data_dir}/ses2spk_${spk}"
output_dir="exp/alignment/${base}_${spk}${suffix}"

# others
log_dir="exp/log/${base}_${spk}${suffix}"
nj=1

. ./path.sh
. ./cmd.sh
. utils/parse_options.sh


for dir in ${graph_dir} ${output_dir} ${log_dir}
do
  [ ! -d ${dir} ] && mkdir -p ${dir}
done

if [ ${stage} -le 0 ]; then
  echo "$0: stage 0, doing BSS"
  if [ ${nj} -le 1 ]; then
    ${alignment_cmd} ${log_dir}/bss.log local/prepare_bss.py ${raw_wav_file} ${tmp_dir} ${bss_output_dir} ${wav_file}
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
      ${bss_output_dir} \
      ${wav_file}
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
    ${lang_dir} \
    ${graph_dir}
fi

if [ ${stage} -le 3 ]; then
  echo "$0: stage 3, doing flexible alignment"
  # to use GPU, use alingment_cmd
  ${alignment_cmd} ${log_dir}/align.log align.py \
   ${model} \
   ${dataset} \
   ${text} \
   ${ses2spk} \
   ${lang_dir} \
   ${graph_dir} \
   ${output_dir} \
   ${use_xvector}
fi

if [ ${stage} -le 4 ]; then
  echo "Debugging"
  utils/int2sym.pl -f 3-4 ${lang_dir}/words.txt ${graph_dir}/G.fst.txt > ${graph_dir}/g_text
  local/concat.py ${output_dir}/text ${output_dir}/text_concat "utt_id"
  compute-wer --text ark:${data_dir}/text_concat_suffix ark:${output_dir}/text_concat
  compute-wer --text --mode=present ark:${data_dir}/text_suffix ark:${output_dir}/text
fi

exit 0;
