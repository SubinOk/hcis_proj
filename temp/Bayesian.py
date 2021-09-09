def black_box_function(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1

from bayes_opt import BayesianOptimization

pbounds = {'x': (2, 4), 'y': (-3, 3)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

logger = JSONLogger(path="./bayesian_logs.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(
    init_points=5,
    n_iter=10,
)

from bayes_opt.util import load_logs

new_optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds={"x": (-2, 2), "y": (-2, 2)},
    verbose=2,
    random_state=7,
)
print(len(new_optimizer.space))

load_logs(new_optimizer, logs=["./bayesian_logs.json"])

print("New optimizer is now aware of {} points.".format(len(new_optimizer.space)))

new_optimizer.maximize(
    init_points=0,
    n_iter=10,
)
