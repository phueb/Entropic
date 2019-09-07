import argparse
import pickle
import socket
import yaml
from datetime import datetime

from init_experiments import config
from init_experiments.job import main
from init_experiments.params import param2requests, param2default

hostname = socket.gethostname()


def run_on_cluster():
    """
    run multiple jobs on multiple LudwigCluster nodes.
    """

    p = config.RemoteDirs.root / '{}_param2val_chunk.pkl'.format(hostname)
    with p.open('rb') as f:
        param2val_chunk = pickle.load(f)
    for param2val in param2val_chunk:

        # check if host is down - do this before any computation
        assert config.RemoteDirs.runs.exists()  # this throws error if host is down

        # execute job
        main(param2val)

        # write param2val to shared drive
        param2val_p = config.RemoteDirs.runs / param2val['param_name'] / 'param2val.yaml'
        print('Saving param2val to:\n{}'.format(param2val_p))
        if not param2val_p.exists():
            param2val['job_name'] = None
            with param2val_p.open('w', encoding='utf8') as f:
                yaml.dump(param2val, f, default_flow_style=False, allow_unicode=True)

    print('Finished all {} jobs at {}.'.format(config.LocalDirs.src.name, datetime.now()))
    print()


def run_on_host():
    """
    run jobs on the local host for testing/development
    """
    from ludwigcluster.utils import list_all_param2vals

    for param2val in list_all_param2vals(param2requests, param2default,
                                         update_d={'param_name': 'param_test', 'job_name': 'job_test'}):
        main(param2val)
        raise SystemExit('Finished running first job.\n'
                         'No further jobs will be run as results would be over-written')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', default=False, action='store_true', dest='local', required=False)
    parser.add_argument('-d', default=False, action='store_true', dest='debug', required=False)
    namespace = parser.parse_args()
    if namespace.debug:
        config.Eval.debug = True
    #
    if namespace.local:
        run_on_host()
    else:
        run_on_cluster()