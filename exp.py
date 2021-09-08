import hashlib
import os

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory.' + directory)

def save_exp_result(manager):
    create_folder('results')
    hash_key = hashlib.sha1().hexdigest()[:6]
    filename = 'results/bayesian_logs-{}.json'.format(hash_key)
    logger = JSONLogger(path=filename)
    manager.bayes_optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

