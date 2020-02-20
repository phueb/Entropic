# Straddler
Research code for understanding lexical category learning in the RNN

## Background

The goal is to understand at what time during training on pseudo-language sequences, does the RNN first create lexical category boundaries. This is investigated by training the RNN on the statistical behavior of pseudo-words belonging to two lexical categories, and noting when knowledge about the difference between these two categories pushes the representation of a straddler, a word that is an equally good member of both categories, to be more similar to representations for words in one or the other category.

## Usage

To run the default configuration, call `straaddler.job.main` like so:

```python
from straddler.job import main
from straddler.params import param2default

main(param2default)  # runs the experiment in default configuration
```

## Compatibility

Developed on Ubuntu 16.04 using Python3.7
