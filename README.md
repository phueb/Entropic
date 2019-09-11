# InitExperiments
Research code for experimenting with neural network weight initialization to speed subcategory learning


To run the default configuration, call `init_experiments.job.main` like so:

```python
from init_experiments.job import main
from init_experiments.params import param2default

main(param2default)  # runs the experiment in default configuration
