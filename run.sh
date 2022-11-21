#!/usr/bin/bash

# Copyright 2022 Johns Hopkins University (author: Dongji Gao) 

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1
stop_stage=100

base="EC_toy"
# ssl or fbank
feature_type="fbank" 

# single or multi speaker
spk="single"
suffix=""
use_xvector=false

# lang and data setting
oov="<UNK>"
lang_dir="data/lang_bpe_500"
graph_dir="exp/graph/${base}_${spk}${suffix}"

data_dir="data/${base}"
text="${data_dir}/text"
ses2hlg="${data_dir}/ses2spk"

# BSS and VAD setting
do_bss=false
do_vad=false

auth_token=""
raw_wav_file="${data_dir}/raw/wav.scp"
tmp_dir="data/tmp"
bss_output_dir="${data_dir}/bss"
vad_segment_output_dir="${data_dir}"
wav_file="${data_dir}/wav.scp"

# flexible_alignment setting
model="pretraied_model/pretrained.pt"
output_dir="exp/alignment/${base}_${spk}${suffix}"

allow_insertion=false
insertion_weight=0

debug=false

# others
log_dir="exp/log/${base}_${spk}${suffix}"
nj=1

. ./cmd.sh
. utils/parse_options.sh


for dir in "${graph_dir}" "${output_dir}" "${log_dir}"; do
  [ ! -d "${dir}" ] && mkdir -p "${dir}"
done

if "${do_bss}"; then
  log "Doing BSS"
  mkdir -p "${tmp_dir}"
  mkdir -p "${bss_output_dir}"

  if [ ${nj} -le 1 ]; then
    ${cuda_cmd} ${log_dir}/bss.log local/prepare_bss.py ${raw_wav_file} ${tmp_dir} ${bss_output_dir} ${wav_file}
  else
    split_wav=""
    for n in $(seq ${nj}); do
      split_wav="${split_wav} ${tmp_dir}/wav_${n}.scp"
    done
    #utils/split_scp.pl ${wav_file} ${split_wav}
    # TODO: split wav.scp based on whole session
    # TODO: write wav.scp
    ${cuda_cmd} JOB=1:${nj} ${log_dir}/bss_JOB.log local/prepare_bss.py \
      ${tmp_dir}/wav_JOB.scp \
      ${tmp_dir} \
      ${bss_output_dir} \
      ${wav_file}
  fi
else
  [ ! -f "${wav_file}" ] && cp "${raw_wav_file}" "${wav_file}"
  log "Skip doing BSS"
fi

#TODO(Dongji): replace pyannote
if "${do_vad}"; then
  echo "$0: stage 1, doing VAD on audio"
  ${cuda_cmd} ${log_dir}/vad.log local/vad.py \
    --auth-token "${auth_token}" \
    --collar 0.0 \
    ${wav_file} \
    ${vad_segment_output_dir}
else
  log "Skip doing VAD"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Stage 1: preparing lang"
  if [ -f ${lang_dir}/lexicon.txt ]; then
    ./local/prepare_lang.py --lang-dir "${lang_dir}"
  else
    log "Lexicon file (${lang_dir}/lexicon.txt) must be provided"
    exit 1
  fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  log "Stage 2: making alignment graph (G.fst.txt)" 
  ./local/make_g.py \
    --text-file "${text}" \
    --lang-dir "${lang_dir}" \
    --output-dir "${lang_dir}" \
    --allow-insertion "${allow_insertion}" \
    --insertion-weight "${insertion_weight}"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  log "Stage 3: making decoding graph (HLG)" 
  ./local/compile_hlg.py \
    --lang-dir ${lang_dir}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  log "Stage 4: preparing lhotse cutset"
  ${cuda_cmd} "${log_dir}/prepare_lhotse.log" local/prepare_lhotse_cutset.py \
    --data-dir "${data_dir}" \
    --lang-dir "${lang_dir}" \
    --feature-type "${feature_type}"
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  log "Stage 5: doing flexible alignment"
  ${cuda_cmd} "${log_dir}/flexible_alignment.log" \
  ./flexible_alignment.py \
    --data-dir "${data_dir}" \
    --lang-dir "${lang_dir}" \
    --checkpoint "${model}" \
    --exp-dir "${output_dir}"

fi

#  log "Stage 4: doing flexible alignment"
#  ${cuda_cmd} ${log_dir}/align.log align.py \
#   ${model} \
#   ${dataset} \
#   ${text} \
#   ${ses2spk} \
#   ${lang_dir} \
#   ${graph_dir} \
#   ${output_dir} \
#   ${use_xvector}
#fi

#if "${debug}"; then
#  log "Stage 3: debugging"
#  utils/int2sym.pl -f 3-4 ${lang_dir}/words.txt ${graph_dir}/G.fst.txt > ${graph_dir}/g_text
#  local/concat.py ${output_dir}/text ${output_dir}/text_concat "utt_id"
#  compute-wer --text ark:${data_dir}/text_concat_suffix ark:${output_dir}/text_concat
#  compute-wer --text --mode=present ark:${data_dir}/text_suffix ark:${output_dir}/text
#else
#  log "Skip debug (scoring)"
#fi

echo "Done."
exit 0;
