# CS-TextNormalization
We build a pipeline to clean text noisy code-switched text online.

## Project Structure

* `DataManagement`: This folder contains the various abstractions that make up the pipeline. When you add a new implementation of some tool for the pipeline, make sure that it is always along the lines of an abstraction contained in this folder. Feel free to add new abstractions into this folder. Some of the abstractions are as follows:  
    `languageUtils.py`: Classes for Langauge Specific Identifiers, Lexicons and SpellCheckers.  
    `dataloader.py`: Classes for loading a corpus - mono-lingual/multi-lingual.  


## Dataset

Twitter

## Usage

You can use this pipeline end to end, or run the individual components within 

```[bash]
python main.py --source-file="source_tanglish.txt" --lang-set="english,telugu"
```
