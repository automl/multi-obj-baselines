from time import sleep
from threading import Thread
import os
import shutil

import argparse

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from hpbandster.core.nameserver import NameServer
from hpbandster.optimizers.mobohb import MOBOHB
from mobohb_worker import MOBOHBWorker


def main_mobohb():
    res = mobohb.run(n_iterations=10e20)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_MOBOHB(
        experiment,
        search_space,
        num_initial_samples=10,
        num_candidates=24,
        gamma=0.10,
        seed=0,
        num_iterations=2000,
        history_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'history', 'mobohb'),
        init_method='random',
        budget=25,
        min_budget=5,
        max_budget=25,
        init=True,
):

    NS = NameServer(run_id=str(seed), host='127.0.0.1', port=0)
    ns_host, ns_port = NS.start()
    w = MOBOHBWorker(experiment, search_space.as_uniform_space(), None, seed, run_id=str(seed), host='127.0.0.1', nameserver=ns_host, nameserver_port=ns_port)
    w.run(background=True)

    motpe_params = {
        'init_method': init_method,
        'num_initial_samples': num_initial_samples,
        'num_candidates': num_candidates,
        'gamma': gamma,
        'budget': budget
    }
    mobohb = MOBOHB(configspace=search_space.as_uniform_space(), parameters=motpe_params, history_dir=history_dir, init=init,
                  run_id=str(seed), nameserver=ns_host, nameserver_port=ns_port,
                  min_budget=min_budget, max_budget=max_budget
                  )

    main_mobohb = lambda : mobohb.run(n_iterations=num_iterations)

    t = Thread(target=main_mobohb)
    t.daemon = True
    t.start()

    snoozeiness = 24 * 3600
    mobohb.is_write()
    sleep(snoozeiness)

    while mobohb.is_write():
        sleep(2)

    mobohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    return experiment
