#!/usr/bin/env bash


#
cd /home/ph/LudwigCluster/scripts
bash kill_job.sh InitExperiments
#bash reload_watcher.sh

echo "Submitting to Ludwig..."
cd /home/ph/InitExperiments/
source init_experiments_venv/bin/activate
python submit.py -r50 -s
deactivate
echo "Submission completed"

sleep 0
tail -n 10 /media/research_data/stdout/*.out


# watch -n1 "cd /media/research_data/InitExperiments/runs; find .  -name *.csv | wc -l"