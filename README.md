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
### Data preparation
#### text file
##### This file contains the transcriptions of each utterance. Each line is in the form of "UTTERANCE_ID TRANSCRIPTS" where UTTERANCE_ID should be SESSION_SPEAKER_INDEX. Lines must be in the chronological order. For example:
```
session1_speaker1_utterance1 He expired two hours later
session1_speaker2_utterance1 Expired means die right
session1_speaker1_utterance2 Pancreas I I think so
session1_speaker3_utterance1 Ah yes <LAU>
session1_speaker2_utterance2 Okay
```
#### audio file
##### This file contains the wav audio of each speaker. Each line is in the form of " transcription".
