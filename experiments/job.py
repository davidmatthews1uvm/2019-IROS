import random
import sys

import numpy
# optimization package: uses AFPO.
from evodevo.evo_run import EvolutionaryRun
# parallelism package
from parallelpy import parallel_evaluate

sys.path.insert(0, "..")

# import the individual wrapper
from experiments.w2v_robot import W2VRobot

# import the robot morphologies
from experiments.w2v_vecs import *
from experiments.quadruped import Quadruped
from experiments.twig import Twig
from experiments.spherebot import SphereBot

# Attempt to setup parallelism in MPI mode.
# will automatically fall back to a single node no MPI setup if needed.
parallel_evaluate.setup(parallel_evaluate.PARALLEL_MODE_MPI_INTER)


def get_internal_bot():
    return None

robot_factory = None

EVAL_TIME = 500
POP_SIZE = 50
GENS = 6000
MAX_RUNTIME = 12 # 12 hours of walltime.

if __name__ == '__main__':
    assert len(sys.argv) >= 3, "please run as python job.py seed experiment_name"
    seed = int(sys.argv[1])
    name = sys.argv[2]

    if len(sys.argv) >= 4:
        MAX_RUNTIME = float(sys.argv[4])

    if len(sys.argv) >= 5:
        parallel_evaluate.MAX_THREADS = int(sys.argv[5])

    numpy.random.seed(seed)
    random.seed(seed)

    numpy.set_printoptions(suppress=True, formatter={'float_kind': lambda x: '%4.2f' % x})

    # generate commands
    # are we running a Balanced training set or the default one?
    if "Balance" in name:
        forwardTaskTrain = [[forward], [foward]]
        backwardTaskTrain = [[backward], [backwards]]
        stopTaskTrain = [[stop], [cease], [suspend]]

        stopTaskTestCmd = random.choice(stopTaskTrain)
        stopTaskTrain.remove(stopTaskTestCmd)
        stopTaskTest = [stopTaskTestCmd]

        train_cmds = {"forward": forwardTaskTrain, "backward": backwardTaskTrain, "stop": stopTaskTrain}

        test_cmds = {"stop": stopTaskTest}
    else:
        forwardTaskTrain = [[forward]]
        backwardTaskTrain = [[backward]]
        stopTaskTrain = [[stop], [cease], [suspend], [halt]]

        stopTaskTestCmd = random.choice(stopTaskTrain)
        stopTaskTrain.remove(stopTaskTestCmd)
        stopTaskTest = [stopTaskTestCmd]

        train_cmds = {"forward": forwardTaskTrain, "backward": backwardTaskTrain, "stop": stopTaskTrain}
        test_cmds = {"stop": stopTaskTest}


    # select internal robot morphology
    print(name)

    # Override default hidden neuron count?
    hidden_neuron_count = 5
    if "HDN_" in name:
        index = name.find("HDN_")
        hidden_neuron_count = int(name[index + 4: index + 6])

    # select morphology type
    if "Quad" in name:
        include_sensors = not ("No_Sensors" in name)

        def get_internal_bot():
            return Quadruped(sensors=include_sensors, num_hidden_neurons=hidden_neuron_count)

    elif "Twig" in name:
        include_sensors = not ("No_Sensors" in name)

        def get_internal_bot():
            return Twig(sensors=include_sensors, num_hidden_neurons=hidden_neuron_count)

    elif "Ball" in name:
        include_sensors = not ("No_Sensors" in name)
        two_dof = not ("1DOF" in name)


        def get_internal_bot():
            return SphereBot(sensors=include_sensors, second_joint=two_dof, num_hidden_neurons=hidden_neuron_count)

    assert get_internal_bot() != None, "Must select what type of robot we are using!"

    # Select treatment type
    if "Control" in name:

        def shuffle_vec(vec):
            for i in range(len(vec) - 1, 0, -1):
                j = random.randint(0, i)
                tmp = vec[i]
                vec[i] = vec[j]
                vec[j] = tmp


        for task in train_cmds:
            tcmds = train_cmds[task]
            for tcmd in tcmds:
                for vec in tcmd:
                    shuffle_vec(vec)

        for task in test_cmds:
            tcmds = test_cmds[task]
            for tcmd in tcmds:
                for vec in tcmd:
                    shuffle_vec(vec)

    # Setup evo run
    if robot_factory is None:
        def robot_factory():
            internal_robot = get_internal_bot()
            return W2VRobot(internal_robot, train_cmds, test_cmds=test_cmds, eval_time=EVAL_TIME)

    def create_new_job():
        return EvolutionaryRun(robot_factory, GENS, seed, pop_size=POP_SIZE, experiment_name=name, override_git_hash_change=False, max_time=MAX_RUNTIME, run_dir="%s_%d"%(name, seed))


    # run evo run.
    job_name = name + "_run_" + str(seed)
    print("Starting run with %d individuals for %d generations with seed %d" % (POP_SIZE, GENS, seed))
    print("Each robot simulated for %d steps with hdn neurons: %d" % (EVAL_TIME, hidden_neuron_count))

    evolutionary_run = create_new_job()
    evolutionary_run.run_full(printing=True)