# CS-TextNormalization
We build a pipeline to clean text noisy code-switched text online.

## Project Structure

* `DataManagement`: This folder contains the various abstractions that make up the pipeline. When you add a new implementation of some tool for the pipeline, make sure that it is always along the lines of an abstraction contained in this folder. Feel free to add new abstractions into this folder. Some of the abstractions are as follows:
		*`languageUtils.py`: Classes for Langauge Specific Identifiers, Lexicons and SpellCheckers
  *`dataloader.py`: Classes for loading a corpus - mono-lingual/multi-lingual.


## Dataset

Twitter

## Environment

The (pseudo-) template code is written in Python 3.6 using some supporting third-party libraries. We will provide a conda environment to install Python 3.6 with required libraries. Simply run

```[bash]
conda env create -f environment.yml
```

## Usage

You can use this pipeline end to end, or run the individual components within 

```[bash]
python vocab.py --train-src=data/train.de-en.de.wmixerprep --train-tgt=data/train.de-en.en.wmixerprep data/vocab.bin
```

```
python nmt.py decode --beam-size=5 --max-decoding-time-step=100 --embed-size=512 --vocab="data/vocab.bin" "work_dir/model_epoch_17_ppl_10.0793.t7" "data/test.de-en.de" "work_dir/decode.txt"
```



