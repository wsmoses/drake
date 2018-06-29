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
        # One input, one output, two state variables.
        VectorSystem.__init__(self, 2, 2)
        self._DeclareContinuousState(4)

    # qddot(t) = u(t)
    def _DoCalcVectorTimeDerivatives(self, context, u, x, xdot):
        #print(x.shape)
        #print(xdot.shape)
        #print(u.shape)
        xdot[0] = x[2]
        xdot[1] = x[3]
        xdot[2] = u[0]
        xdot[3] = u[1]

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

def get_grid(div, num_q_bins=5, num_qdot_bins=7):
    qbins = np.linspace(-3., 3., (int(num_q_bins / div) / 2) * 2 + 1)
    qdotbins = np.linspace(-3., 3., (int(num_qdot_bins / div) / 2) * 2 + 1)
    state_grid = [set(qbins), set(qbins), set(qdotbins), set(qdotbins)]
    return state_grid

input_limit = 1.
input_grid = [set(np.linspace(-input_limit, input_limit, 7)), set(np.linspace(-input_limit, input_limit, 7))]
timestep = 0.01

def interpolator(data, newsize):
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(tuple(np.arange(s) for s in data.shape), data)
    nx = np.indices(newsize).astype(np.float64)
    for i, s in enumerate(newsize):
        nx[i] *= (data.shape[i] - 1)/ float(s - 1)
    nx = np.moveaxis(nx, 0, len(newsize))
    return interp(nx)

import sys

def run(q_bins=9, qdot_bins=11, levels=3):
    grid = get_grid(1<<(levels-1), q_bins, qdot_bins)
    shape = tuple(map(len, grid))[::-1]
    #print(shape)
    policy, cost_to_go, pi, runtime = init_alg(simulator, cost_function,
                                              grid, input_grid,
                                              timestep, options)
    #print cost_to_go.shape, shape, (q_bins, qdot_bins)
    sys.stdout.flush()
    cost_to_go = np.reshape(cost_to_go, shape)
    pi = np.reshape(pi, (pi.shape[0],) + shape)

    print "I,"+str(shape)+","+str(runtime)
    sys.stdout.flush()

    #print levels
    for level in range(levels-2, -1, -1):
        #print level
        grid = get_grid(1<<level, q_bins, qdot_bins)
        shape = tuple(map(len, grid))[::-1]
        #print shape
        #sys.stdout.flush()
        cost_to_go = interpolator(cost_to_go, shape)
        cost_to_go.resize(cost_to_go.size)
        pi = interpolator(pi, (pi.shape[0],)+shape)
        pi.resize((pi.shape[0], pi.size/pi.shape[0]))

        #print "V", pi.shape, cost_to_go.shape
        #sys.stdout.flush()
        def tmp():
            policy, cost_to_go, pi, runtime = init_alg(simulator, cost_function,
                                          grid, input_grid,
                                          timestep, options)
            print "T", pi.shape, cost_to_go.shape
            sys.stdout.flush()
        #tmp()

        policy, cost_to_go, pi, runtime = resume_alg(simulator, cost_function,
                                                  grid, input_grid,
                                                  timestep, cost_to_go, pi, options)

        cost_to_go = np.reshape(cost_to_go, shape)
        pi = np.reshape(pi, (pi.shape[0],)+shape)

        print "R,"+str(shape)+","+str(runtime)
        sys.stdout.flush()


for i in range(0, 2000, 2):
    run(9+i, 11+i, 1)
    run(9+i, 11+i, 3)
