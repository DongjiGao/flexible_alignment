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
### **[Deprecated]** <s>Install [Hugging Face](https://huggingface.co/) for fine-tuning and loading ASR model</s>
```
pip install transformers
pip install datasets
```
### **[Optional]** Install [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) for Blind Source Separation
```
pip install pyroomacoustics
```
### **[Optional]** Install [pyannote](https://pyannote.github.io/) for Voice Activity Detection
```
pip install pyannote.audio
```
Note: you may need token to download pretrained VAD model from Huggingface. For more details, please read [this](https://github.com/pyannote/pyannote-audio).
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
#### lexicon/BPE lexicon
##### This file maps phonemes to word
```
wreck   R EH K
a       AX
nice    N AY S
beach   B IY CH
```
#### ses2hlg
##### The mapping between session and HLG WFST graph
```
session1 0
session2 1
```
#### ASR model and phoneme vocabulary
###### These can be download here.

## Usage
```
./run.sh
```
### Blind Source Serapation
```
${alignment_cmd} ${log_dir}/bss.log local/prepare_bss.py ${raw_wav_file} ${tmp_dir} ${bss_output_dir} ${wav_file}
```
#### This step do blind source separtion on input channels in raw_wav_file (wav.scp). The BSSed audio is stored in bss_output_dir and its wav.scp in wav_file.

### Voice Activity Detection
```
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
```
#### This step do VAD on BSSed audio (wav_file) and write segments in vad_output_dir.
### Stage 1: make lang directory
```
if [ -f ${lang_dir}/lexicon.txt ]; then
  ./local/prepare_lang.py --lang-dir "${lang_dir}"
else
  log "Lexicon file (${lang_dir}/lexicon.txt) must be provided"
  exit 1
fi
```
#### This step makes lexicon WFST (L.fst) in graph_dir

### Stage 2: make flexible alignment graph (G.fst.txt)
```
./local/make_g.py \
  --text-file "${text}" \
  --lang-dir "${lang_dir}" \
  --output-dir "${lang_dir}" \
  --allow-insertion "${allow_insertion}" \
  --insertion-weight "${insertion_weight}"
```
### Stage 3: compile decoding graph (HLGs.pt)
```
./local/compile_hlg.py \
  --lang-dir ${lang_dir}
```
### Stage 4: prepare lhotse dataset and compute acoustic features [ssl,fbank]
```
${cuda_cmd} "${log_dir}/prepare_lhotse.log" local/prepare_lhotse_cutset.py \
  --data-dir "${data_dir}" \
  --lang-dir "${lang_dir}" \
  --feature-type "${feature_type}"
```    
## Results
### ASR model fine-tuned on different pre-trained model 

|| Wav2Vec2-XLSR-53  | Wav2Vec2-Large-Robust|
| --------------- | :---------------:| :---------------: |
|WER | 56 | 43 |

### flexible alignment (WER)
 
<table>
  <tr>
    <td style="text-align:center;" colspan="3">NTU collaborative dataset</td>
  </tr>
  <tr>
    <td>alignment</td>
    <td>flexible alignment</td>
    <td>flexible alignment + xvector rescoring</td>
  </tr>
   <tr>
    <td>15.45</td>
    <td>12.05</td>
    <td>11.31</td>
  </tr>
</table>
