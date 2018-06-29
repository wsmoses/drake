import math
import numpy as np

from pydrake.systems.framework import VectorSystem
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import (
    DynamicProgrammingOptions, FittedValueIteration, TimedFittedValueIteration, ParallelFittedValueIteration,
    ResumedFittedValueIteration)

init_alg = ParallelFittedValueIteration
resume_alg = ResumedFittedValueIteration

class DoubleIntegrator(VectorSystem):
    def __init__(self):
        VectorSystem.__init__(self, 1, 1)
        self._DeclareContinuousState(2)

    # qddot(t) = u(t)
    def _DoCalcVectorTimeDerivatives(self, context, u, x, xdot):
        xdot[0] = x[1]
        xdot[1] = u

    # y(t) = x(t)
    def _DoCalcVectorOutput(self, context, u, x, y):
        y[:] = x

# Set up a simulation of this system.
plant = DoubleIntegrator()
simulator = Simulator(plant)
options = DynamicProgrammingOptions()

# This function evaluates a running cost
# that penalizes distance from the origin,
# as well as control effort.
def quadratic_regulator_cost(context):
    x = context.get_continuous_state_vector().CopyToVector()
    u = plant.EvalVectorInput(context, 0).CopyToVector()
    return 2*x.dot(x) + 10*u.dot(u)

# Pick your cost here...
#cost_function = min_time_cost
cost_function = quadratic_regulator_cost

def get_grid(div, num_q_bins=131, num_qdot_bins=151):
    qbins = np.linspace(-3., 3., int(int(num_q_bins / div) / 2) * 2 + 1)
    qdotbins = np.linspace(-3., 3., int(int(num_qdot_bins / div) / 2) * 2 + 1)
    state_grid = [set(qbins), set(qdotbins)]
    return state_grid

input_limit = 1.
input_grid = [set(np.linspace(-input_limit, input_limit, 9))]
timestep = 0.01


def interpolator(data, newsize):
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(tuple(np.arange(s) for s in data.shape), data)
    nx = np.indices(newsize).astype(np.float64)
    for i, s in enumerate(newsize):
        nx[i] *= (data.shape[i] - 1)/ float(s - 1)
        nx[i] = np.clip(nx[i], 0, (data.shape[i] - 1))
    nx = np.moveaxis(nx, 0, len(newsize))
    return interp(nx)

import sys

def run(q_bins=131, qdot_bins=151, levels=3):
    grid = get_grid(1.1**(levels-1), q_bins, qdot_bins)
    policy, cost_to_go, pi, runtime = init_alg(simulator, cost_function,
                                              grid, input_grid,
                                              timestep, options)

    cost_to_go = np.reshape(cost_to_go, (len(grid[1]), len(grid[0])))
    pi = np.reshape(pi, (len(grid[1]), len(grid[0])))

    print "I,"+str((len(grid[1]), len(grid[0])))+","+str(runtime)
    sys.stdout.flush()

    for level in range(levels-2, -1, -1):
        grid = get_grid(1.1**level, q_bins, qdot_bins)

        cost_to_go = interpolator(cost_to_go, (len(grid[1]), len(grid[0])))
        cost_to_go.resize(cost_to_go.size)
        pi = interpolator(pi, (len(grid[1]), len(grid[0])))
        pi.resize((1, pi.size))

        policy, cost_to_go, pi, runtime = resume_alg(simulator, cost_function,
                                                  grid, input_grid,
                                                  timestep, cost_to_go, pi, options)

        cost_to_go = np.reshape(cost_to_go, (len(grid[1]), len(grid[0])))
        pi = np.reshape(pi, (len(grid[1]), len(grid[0])))

        print "R,"+str((len(grid[1]), len(grid[0])))+","+str(runtime)
        sys.stdout.flush()


for i in range(0, 2000, 50):
    run(131+i, 151+i, 1)
    run(131+i, 151+i, 3)
