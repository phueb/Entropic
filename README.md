<div align="center">
 <img src="images/logo.png" width="250">
</div>


Research code for understanding lexical category learning in the RNN.

## Background

### Learning from non-stationary input

Infants learn lexical subcategory structure in an online fashion, that is, from language that changes as they age.
Non-stationary is problematic, if positing lexical category acquisition as an RNN-like mechanism predicting upcoming words.
Having been exposed to one category structure during early training, the RNN cannot incorporate new examples,
 without re-organizing existing knowledge (also known as interference). 
For example, how does the RNN learn new semantic sub-divisions of the noun category, 
  when it has already committed to a particular noun category structure given a set of early training examples?

### How to minimize interference?

The period abstracts the noun-category, making it coherent, pulling nouns together in representational space. 
But it also makes the left-contexts of nouns useful predictors of the noun category. 
These left-contexts become useful later when new nouns are seen, because it makes them differentiate via the noun-category,
 and not via some higher superordinate category. 
An unseen word, in the absence of any context would be considered by the RNN to be part of the largest, most entropic category,
 and this assumption would be wrong for nouns. So, when new nouns occur in good noun contexts (learned early),
 they are differentiated via the noun-category, and their trajectory through representational space mimics nouns in general. 
This explains where the extra interference comes from when training in reverse age-order: Not having learned useful noun-contexts,
 when the RNN sees new nouns, it does not differentiate them with respect to the noun category,
 but with respect to some more entropic category, which is higher up in the subcategory tree that the RNN has learned to encode.
Because differentiation starts higher up in the tree, this affects a larger number of learned representations.
Rather than only having to re-organize nouns, the RNN is force to re-organize many more words, when a new noun is encountered.


### Learning Dynamics

#### Without pseudo-periods

The animation below illustrates the differentiation of 4 categories by the RNN. 
Each category word is followed by mutually exclusive set of next-words.

<div align="center">
 <img src="images/pp=0.0_output_probs.gif" width="600">
</div>

#### With pseudo-periods

In another simulation, each category word co-occurred with a pseudo-period with probability=0.1.
A pseudo-period is a word that indiscriminately follows words from any category.
Training on a corpus with pseudo-words included, changes the learning dynamics in an important way:
Before the 4 categories differentiate, their representations __first converge onto a common reference point__.

<div align="center">
 <img src="images/pp=0.1_output_probs.gif" width="600">
</div>

### Implications

One previously unexplained finding training the RNN on artificial language sequences is that the RNN converges faster on the target categories if there are fewer of them.
Using the principle of maximum entropy and knowledge of temporarily stable "phantom" categories, this can be explained as follows:
The fewer target categories the RNN is tasked to learn, the fewer temporarily stable "phantom" states must be traversed during training.


Moreover, "phantom" categories may help researchers understand how to train RNN language models that remain in a more uncommitted/undifferentiated state longer.
Staying uncommitted longer during early training should help category learning in situations where the input distribution is non-stationary,
such as training on child-directed speech in the order in which it is actually experienced by children on a developmental time-scale (1-6 first years of life).

## Usage

To run the default configuration, call `entropic.job.main` like so:

```python
from entropic.job import main
from entropic.params import param2default

main(param2default)  # runs the experiment in default configuration
```

## Compatibility

Developed on Ubuntu 16.04 using Python3.7
