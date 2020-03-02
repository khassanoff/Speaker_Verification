# PyTorch_Speaker_Verification

PyTorch Implementation of the model described in UTTERANCE-LEVEL AGGREGATION FOR SPEAKER RECOGNITION IN THE WILD: https://arxiv.org/pdf/1902.10107.pdf


# Dependencies

Create the conda environment:
```
conda env create -f environment.yml 
conda activate SV_pytorch
```


# Preprocessing
Download the Vox Celeb dataset from here: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html

Once you download the dataset specify the proper path instead of PATHTODATASET in config/config.yml:

```yaml
train_path: './PATHTODATASET/Audio/dev/processed'
train_path_unprocessed: '/PATHTODATASET/Audio/dev/wav' #*/wav/speaker_id/session_id/file.wav
test_meta_path: './dataVoxCeleb1/Metadata/veri_test.txt'
test_path: '/PATHTODATASET/Audio/test/processed'
test_path_unprocessed: '/PATHTODATASET/Audio/test/wav'


```

To convert the wav files to spectograms run the following utility function:
```
python data_preprocess.py
```

Two folders will be created, dev/processed and test/processed, containing .npy files containing numpy ndarrays of speaker utterances and test pair utterances, respectively. 


# Training

To train the speaker verification model, run:
```
python train_speech_embedder.py 
```
with the following config.yaml key set to true:
```yaml
training: !!bool "true"
```
for testing, set the key value to:
```yaml
training: !!bool "false"
```
The log file and checkpoint save locations are controlled by the following values:
```yaml
log_file: './speech_id_checkpoint/Stats'
checkpoint_dir: './speech_id_checkpoint'
```

