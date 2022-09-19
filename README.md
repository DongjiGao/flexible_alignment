# Flexible Alignment <img src="https://raw.githubusercontent.com/k2-fsa/k2/master/docs/source/_static/logo.png" width=44>

<div align="center">
  <img src="https://github.com/DongjiGao/flexible_alignment/blob/master/figures/model.png" width=800>
</div>

## Requirements
### Install [k2](https://k2-fsa.github.io/k2/), [Lhotse](https://github.com/lhotse-speech/lhotse), [Icefall](https://github.com/k2-fsa/icefall) for building WFST
```
conda install -c k2-fsa -c pytorch -c conda-forge k2 python=3.8 cudatoolkit=11.3 pytorch=1.11
pip install git+https://github.com/lhotse-speech/
git clone https://github.com/k2-fsa/icefall
cd icefall
pip install -r requirements.txt
export PYTHONPATH=/tmp/icefall:$PYTHONPATH
```
### Install [Hugging Face](https://huggingface.co/) for fine-tuning and loading ASR model
```
pip install transformers
pip install datasets
```
### Install [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) for blind source separation
```
pip install pyroomacoustics
```
### Install [pyannote](https://pyannote.github.io/) for voice activity detection
```
https://pyannote.github.io/
```
## Usage
```
./run.sh
```
## Data preparation
#### text file
##### This file contains the transcriptions of each utterance. Each line is in the form of 
```
"<utterance_id> <transcription>" 
```
##### and the format of \<utterance_id\> is 
```
<session_id>_<speaker_id>_<utterance_id_of_speaker>
```
##### Lines must be in the chronological order. 
##### For example:
```
session1_speaker1_utterance1 He expired two hours later
session1_speaker2_utterance1 Expired means die right
session1_speaker1_utterance2 Pancreas I I think so
session1_speaker3_utterance1 Ah yes <LAU>
session1_speaker2_utterance2 Okay
```
#### audio file
##### This file contains the location of audio for each speaker. Each line is in the form of
```
<recording-id> <location_of_audio_file>
```
##### and the format of \<recording_id\> is 
```
<session_id>_<speaker_id>
```
##### For example:
```
session1_speaker1 /DATA/session1_speaker1.wav
session1_speaker2 /DATA/session1_speaker2.wav
session1_speaker3 /DATA/session1_speaker3.wav

```
#### lexicon
##### This file maps phonemes to word
```
wreck   R EH K
a       AX
nice    N AY S
beach   B IY CH
```
#### ses2spk
##### The mapping between session and speakers
```
session1 speaker1
session1 speaker2
session2 spkeaer3
session2 spkeaer4
```
#### ASR model and phoneme vocabulary
###### These can be download here.

## Usage
```
./run.sh
```
### Stage 0: blind source serapation
```
${alignment_cmd} ${log_dir}/bss.log local/prepare_bss.py ${raw_wav_file} ${tmp_dir} ${bss_output_dir} ${wav_file}
```
#### This step do blind source separtion on input channels in raw_wav_file (wav.scp). The BSSed audio is stored in bss_output_dir and its wav.scp in wav_file.

### Stage 1: voice activity detection
```
run.pl ${log_dir}/vad.log local/vad.py \
  --ref-segment "data/ntu/segments" \
  --collar 0.2 \
  --gap 0.5 \
  --metric "precision_recall" \
  ${wav_file} \
  ${vad_output_dir}
```
#### This step do VAD on BSSed audio (wav_file) and write segments in vad_output_dir. If ground truth segment is provided, it can analyze the quality of VAD results by measuring [precision_call, IoU (intersection over union), Jaccard error rate].

### Stage 2: make lang directory
```
local/prepare_data.sh \
  --oov ${oov} \
  ${lang_dir} \
  ${graph_dir}
```
#### This step makes lexicon WFST (L.fst) in graph_dir

### Stage 3: flexible alignment
```
${alignment_cmd} ${log_dir}/align.log align.py \
  ${model} \
  ${dataset} \
  ${text} \
  ${ses2spk} \
  ${lang_dir} \
  ${graph_dir} \
  ${output_dir} \
  ${use_xvector}
```
#### This step does 3 things: 
  1) build decoding (alignment graph) of given text. 
  2) integrate ASR mode and decoding graph in graph_dir (aligner) 
  3) do flexible alingment for dataset. 
  if use_xvector is turned on
  4) get nbest alignment and do xvector rescoring (speaker dependent)

## Results
### ASR model
