# CS-TextNormalization
We build a pipeline to clean text noisy code-switched text online.

## Getting the repo

`git clone --recursive https://github.com/sumeet-iitg/CS-TextNormalization.git`

-- Don't miss the 'recursive' part for pulling required sub-modules

## Project Structure

* `DataManagement`: This folder contains the various abstractions that make up the pipeline. When you add a new implementation of some tool for the pipeline, make sure that it is always along the lines of an abstraction contained in this folder. Feel free to add new abstractions into this folder. Some of the abstractions are as follows:  
    `languageUtils.py`: Classes for Langauge Specific Identifiers, Lexicons and SpellCheckers.  
    `dataloader.py`: Classes for loading a corpus - mono-lingual/multi-lingual.  

## Requirements

- [fuzzy](https://pypi.org/project/Fuzzy/)
- [pybktree](https://pypi.org/project/pybktree/)
- [indictrans](https://pypi.org/project/indic-transliteration/)
- [codecs](https://pypi.org/project/openapi-codec/)
- [ekphrasis](https://github.com/cbaziotis/ekphrasis)
- [emoji](https://pypi.org/project/emoji)
- [bs4](https://pypi.org/project/beautifulsoup4/)
- [metaphone](https://pypi.org/project/Metaphone/)
## Usage

You can use this pipeline end to end, or run the individual components within 

```[bash]
python main.py "source_tanglish.txt" "english,telugu"
```
