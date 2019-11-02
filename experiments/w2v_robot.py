import math
import uuid
import os

import numpy as np
from scipy import spatial

from parallelpy.utils import Work, Letter
from Pyrosim.pyrosim import pyrosim
from evodevo.moo_interfaces import MOORobotInterface


class W2VRobot(MOORobotInterface):
    def __init__(self, robot, cmds, eval_time=500, quasi_static_ratio=1, test_cmds=None):
        self.id = -1
        self.parent_id = -1
        self.robot = robot
        self.train_commands = cmds
        self.num_train_cmds = len(self._flatten(self.train_commands.values()))

        self.test_commands = test_cmds
        if self.test_commands is None:
            self.num_test_commands = 0
        else:
            self.num_test_commands =len(self._flatten(self.test_commands.values()))

        self.ttl_num_cmds = self.num_train_cmds + self.num_test_commands

        self.fitness = ({}, {})  # (train, test)
        # self.behavioral_sem_error = np.zeros(shape=(self.ttl_num_cmds, self.ttl_num_cmds))

        self.eval_time = eval_time
        self.quasi_static_ratio = quasi_static_ratio
        self.needs_eval = True
        self.play_blind = True
        self.debug = False

        # check if we are running on the vacc. if so, force play_blind = True
        if (os.getenv("VACC") != None):
            self.play_blind = True
        self.play_paused = False
        self.age = 0

    def __str__(self):
        return "ID: %d, PID: %d, age: %d, f: %.2f, f : %.2f, %s %s"% (self.get_id(), self.get_parent_id(), self.get_age(), self.get_fitness(test=False), self.get_fitness(test=True),
                                                                      ["%.2f"%d for d in self._flatten(self.fitness[0].values())], ["%.2f"% d for d in self._flatten(self.fitness[1].values())])

    def __repr__(self):
        return str(self)

    # Methods for MOORObotInterface class

    def set_id(self, new_id):
        self.id = new_id

    def get_id(self):
        return self.id

    def get_parent_id(self):
        return self.parent_id

    def iterate_generation(self):
        self.age += 1

    def needs_evaluation(self):
        return self.needs_eval

    def mutate(self):
        self.parent_id = self.get_id()
        self.needs_eval = True
        self.robot.mutate()
        self.fitness = ({}, {})

    def get_minimize_vals(self):
        return [self.get_age()]

    def get_maximize_vals(self):
        return [self.get_fitness()]


    def get_fitness(self, test=False):
        ret = np.sum(list(self._flatten(self.fitness[0].values())))
        if test:
             ret += np.sum(list(self._flatten(self.fitness[1].values())))

        if np.isnan(ret) or np.isinf(ret) or ret > 30:
            return 0
        else:
            return ret       

    def get_summary_sql_columns(self):
        base =  "(id INT, parentID INT, age INT, fitness FLOAT"
        cmds_to_add = []
        for cmd in sorted(self.train_commands.keys()):
            for cmd_idx in range(len(self.train_commands[cmd])):
                cmds_to_add += ["Train_%s_%d"%(cmd, cmd_idx)]
        for cmd in sorted(self.test_commands.keys()):
            for cmd_idx in range(len(self.test_commands[cmd])):
                cmds_to_add += ["Test_%s_%d" % (cmd, cmd_idx)]
        if cmds_to_add:
            base += ", "
            base += ', '.join(["%s FLOAT"%cmd for cmd in cmds_to_add])
        base += ")"
        return base

    def get_summary_sql_data(self):
        to_ret = (self.get_id(), self.get_parent_id(), self.get_age(), self.get_fitness())
        to_add = []
        for cmd in sorted(self.train_commands.keys()):
            cmd_fitnesses = self.fitness[0][cmd]
            to_add += list(cmd_fitnesses)
        for cmd in sorted(self.test_commands.keys()):
            cmd_fitnesses = self.fitness[1][cmd]
            to_add += list(cmd_fitnesses)
        to_ret += tuple(to_add)
        return to_ret

    def dominates_final_selection(self, other):
        return self.get_fitness() > other.get_fitness()

    # Methods for Work class
    def cpus_requested(self):
        return 1

    def compute_work(self, test=True, **kwargs):
        if "serial" in kwargs and kwargs["serial"]:
            serial = True
        else:
            serial = False

        sims = self.get_simulator_instances(test=test)
        sims_dat = ({}, {})  # (train, test)

        if serial or True:
            for i in [0,1]:
                for val in sims[i]:
                    if val not in sims_dat[i]:
                        sims_dat[i][val] = [None]*len(sims[i][val])
                    for n, sim in enumerate(sims[i][val]):
                        sims[i][val][n].start()
                        sims_dat[i][val][n] = sims[i][val][n].wait_to_finish()
                        print(".", end="", flush=True)
        self.evaluate_via_sim_data(sims_dat)
        # print(self.fitness)

    def write_letter(self):
        # print("writing letter")
        return Letter(self.fitness, None)

    def open_letter(self, letter):
        # print("opening")
        self.fitness = letter.get_data()
        self.needs_eval = False
        return None



    def get_num_evaluations(self, test=False):
        if test:
            return self.ttl_num_cmds
        else:
            return self.num_train_cmds

    def get_simulator_instances(self, test=False):
        """
        Generates and returns all the simulations that need to be run to evaluate this robot.
        :return: An array of simulations to run.
        """
        if "play_paused" not in self.__dict__:
            self.play_paused = True

        if "debug" not in self.__dict__:
            self.debug = False

        if "quasi_static_ratio" not in self.__dict__:
            self.quasi_static_ratio = 1

        if type(self.train_commands) in [list, tuple]:
            self.fitness = ({}, {})
            tmp = self.train_commands
            self.train_commands = {}
            self.train_commands["forward"] = [tmp[0]]
            self.train_commands["backward"] = [tmp[1]]
            self.train_commands["stop"] = list(tmp[2:])

            tmp = self.test_commands
            self.test_commands = {}
            self.test_commands["stop"] = list(tmp)

        start_idx = 0

        if test:
            end_idx = self.ttl_num_cmds
        else:
            end_idx = self.num_train_cmds
        sims = ({}, {})  # (train, test)
        if self.train_commands is not None:
            for val in self.train_commands:
                if val not in sims[0]:
                    sims[0][val] = []
                for cmd in self.train_commands[val]:
                    sim = pyrosim.Simulator(debug=self.debug, eval_time=self.eval_time, play_blind=self.play_blind,
                                      play_paused=self.play_paused, quasi_static_ratio=self.quasi_static_ratio)
                    sims[0][val].append(sim)
                    self.robot.send_to_simulator(sim, cmd)

        if self.test_commands is not None:
            for val in self.test_commands:
                if val not in sims[1]:
                    sims[1][val] = []
                for cmd in self.test_commands[val]:
                    sim = pyrosim.Simulator(debug=self.debug, eval_time=self.eval_time, play_blind=self.play_blind,
                                      play_paused=self.play_paused)
                    sims[1][val].append(sim)
                    self.robot.send_to_simulator(sim, cmd)
        return sims

    def evaluate_via_sim_data(self, sims_dat, test=False):
        for i in [0,1]:
            for val in sims_dat[i]:
                for sim_dat in sims_dat[i][val]:
                    motion_penalty = 1
                    try:
                        motion_penalty = self.robot.get_motion(sim_dat)
                    except Exception as e:
                        print(e)
                        pass
                    x_pos = sim_dat[-1:, 0:1, -1:][0][0][0]
                    y_pos = sim_dat[-1:, 1:2, -1:][0][0][0]

                    tmp = sim_dat[-1:, 0:1, :].flatten()
                    x_delta = tmp[1:] - tmp[:-1]

                    tmp = sim_dat[-1:, 1:2, :].flatten()
                    y_delta = tmp[1:] - tmp[:-1]

                    deltas = list([math.sqrt(x ** 2 + y ** 2) for x, y in zip(x_delta, y_delta)])
                    if (np.max(np.array(deltas).flatten()) > 0.1):
                        fit = 0
                    elif val == "forward":
                        fit = x_pos
                    elif val == "backward":
                        fit = -1 * x_pos
                    elif val == "stop":
                        fit = -1 * np.sum(np.array(deltas))
                    else:
                        raise Exception ("No fitness function for the given command.")
                    if val not in self.fitness[i]:
                        self.fitness[i][val] = []

                    self.fitness[i][val].append(fit / motion_penalty)


    #
    # def calc_error(self, target_sim, behavioral_sim):
    #     """
    #     Calculates and returns semantic behavioral error
    #     :param target_sim: Target behavioral similarity
    #     :param behavioral_sim: Behavioral Similarity
    #     :return: difference between target and behavioral (positive)
    #     """
    #     return abs(target_sim-behavioral_sim)

    #
    #
    # def get_cos_sim(self, v1, v2):
    #     """
    #     calculates and returns the cosine similarity between two numpy arrays.
    #     :param v1: A numpy array
    #     :param v2: A numpy array.
    #     :return: cosine similarity between v1 and v2.
    #     """
    #     v1 = np.array(v1)
    #     v2 = np.array(v2)
    #     return 1 - spatial.distance.cosine(v1.flatten(), v2.flatten())
    #

    def get_age(self):
        return self.age

    def _flatten(self, l):
        ret = []
        for items in l:
            ret += items
        return ret

