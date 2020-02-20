# Straddler
Research code for understanding lexical categlry learning in the RNN


To run the default configuration, call `straaddler.job.main` like so:

```python
from init_experiments.job import main
from init_experiments.params import param2default

main(param2default)  # runs the experiment in default configuration
