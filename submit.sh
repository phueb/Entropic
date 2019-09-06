#!/usr/bin/env bash


#
cd /home/ph/LudwigCluster/scripts
bash kill_job.sh InitExperiments
#bash reload_watcher.sh

echo "Submitting to Ludwig..."
cd /home/ph/InitExperiments/
source venv/bin/activate
python submit.py -r10 -s
deactivate
echo "Submission completed"

sleep 5
tail -n 6 /media/research_data/stdout/*.out