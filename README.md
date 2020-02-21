# Entropic

Research code for understanding lexical category learning in the RNN.

## Background

### Intermediate categories

The goal is to demonstrate that an RNN language model first represents members of different lexical categories as equally good members of emergent superordinate categories,
that are formed during early stages of training.
These superordinate categories may map on to real categories.
They are important as they constrain the representational trajectory of word representations, and help us understand how RNN converges on the target categories.
The composition of early emergent superordinate categories is best understood within the framework of maximum entropy. 
A superordinate category in the RNN can emerge temporarily when a learned output probability distribution best captures a collection of individual next-word probability distributions,
 given the information gathered so far.
In this case, the cross entropy between a single next-word probability distribution is sufficient to capture the next-word probability distributions of all category members. 
As more information is gathered, evidence for differences in next-word probability distributions causes the single next-word probability distribution to no longer be optimal, and the category is split,
to reach optimality again.
This process continues, until the cross entropy is as small as it can be, and no further categories must be split.

### Hypothesis

Learning fewer categories takes fewer weight updates to converge to perfect categorization accuracy.
One explanation is that the fewer target categories the RNN is tasked to learn, the fewer intermediate superordinate categories must be acquired and discarded.
This sequence of acquisition and discarding may slow down learning when the number of target categories is large.

To demonstrate the presence of intermediate, emergent superordinate categories, one must track the evolution of learned next-word probability distributions.
The presence of an emergent category is confirmed if a word's learned next-word probability distribution includes probability density for words that never actually followed the word of interest during training.

## Usage

To run the default configuration, call `entropic.job.main` like so:

```python
from entropic.job import main
from entropic.params import param2default

main(param2default)  # runs the experiment in default configuration
```

## Compatibility

Developed on Ubuntu 16.04 using Python3.7
