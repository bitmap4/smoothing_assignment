# iNLP Assignment 1

Language modeling implementation with various smoothing techniques and text generation capabilities.

## Overview

This project implements:
- Text tokenization
- N-gram language models
- Multiple smoothing techniques (Laplace, Good-Turing, Interpolation)
- Text generation using the trained models

## Directory Structure

```
.
├── models/                   # Modules for language models
│   ├── base.py               # Base class for n-gram models
│   ├── laplace.py            # Laplace smoothing
│   ├── good_turing.py        # Good-Turing smoothing
│   └── interpolation.py      # Interpolation smoothing
├── src/
│   ├── language_model.py     # Language model
│   ├── generator.py          # Text generation
│   └── tokenizer.py          # Tokenization implementation
├── data/
│   ├── pride.txt
│   └── ulysses.txt
├── results/                  # Analysis results
├── analysis.sh               # Analysis script
├── Report.pdf
└── README.md
```

## Pre-requisites

- Python 3
- Numpy

```bash
pip install numpy
```

## Usage

### Tokenizer

Run the tokenizer:
```bash
python3 src/tokenizer.py
```

Example:
```
your text: Is this what you mean? I am unsure.
[['Is', 'this', 'what', 'you', 'mean'], ['I', 'am', 'unsure']]
```

### Language Model

Print average perplexity and perplexities of each sentence:
```bash
python3 src/language_model.py --analyze {train,test} [--method {l,g,i}] [-n N] corpus_path
```
Example:
```bash
python3 src/language_model.py --analyze train data/pride.txt
```

Check sentence probability:
```bash
python3 src/language_model.py [--method {l,g,i}] [-n N] corpus_path
```
- l: Laplace smoothing (default)
- g: Good-Turing smoothing 
- i: Interpolation smoothing

n: N-gram size (default = 3)

Example:
```bash
python3 src/language_model.py -m i -n 3 data/pride.txt
```

### Generator

Generate text continuations:
```bash
python3 src/generator.py [-h] [-m {i,g,l}] [-n N] [-k K] corpus_path
```
- l/g/i: Smoothing type (optional)
- n: N-gram size (default = 3)
- k: Number of candidates to generate (default = 3)

Examples:
```bash
python3 src/generator.py --method i data/pride.txt # with interpolation
python3 src/generator.py -n 1 -k 5 data/ulysses.txt # without smoothing or interpolation
```

### Analysis

Run the analysis script:
```bash
chmod +x analysis.sh
./analysis.sh
```
This will generate results for all three smoothing techniques and all three corpora with n=1,3,5 in the `results/` directory.

## Implementation Details

### Tokenization
- Handles URLs, hashtags, mentions, numbers, punctuation
- Sentence boundary detection
- Special token handling (\<UNK>, \<s>, \</s>, \<NUM>, \<URL>, \<HASHTAG>, \<MENTION>, \<MAILID>)

### Language Models
- Implements unigram, trigram and 5-gram models
- Supports three smoothing techniques
- Calculates perplexity scores

### Generation
- Returns top-k likely continuations
- Handles out-of-vocabulary words