import math
import random
import numpy


class SphereBot(object):
    def __init__(self, sensors=False, num_hidden_neurons=5, second_joint=True):
        self.second_joint=second_joint
        self.num_hidden_neurons = num_hidden_neurons

        # synapses. Initialize to random weights on a normal distribution with a mean of and a standard deviation of 1.
        if sensors:
            self.h_synapses = numpy.random.normal(0, 1, (self.num_hidden_neurons, self.num_hidden_neurons + 2))
            self.m_synapses = numpy.random.normal(0, 1,
                                                  (2, self.num_hidden_neurons))  # synapses connecting to motor neurons
            self.num_sensors = 1
        else:
            self.h_synapses = numpy.random.normal(0, 1, (self.num_hidden_neurons, self.num_hidden_neurons + 1))
            self.m_synapses = numpy.random.normal(0, 1,
                                                  (2, self.num_hidden_neurons))  # synapses connecting to motor neurons
            self.num_sensors = 0

        self.sensor_neurons = {}  # key is neuron number index at 0, value is pyrosim neuron id

        self.hidden_neurons_state = [(0,
                                      0)] * self.num_hidden_neurons  # array of tuples storing (last_value, value) of each hidden neuron.
        self.hidden_neurons = {}  # key is neuron number index at 0, value is pyrosim neuron id

        self.motor_neurons = {}  # key is neuron number index at 0, value is pyrosim neuron id

    def is_exploading(self, sim_dat):
        return False

    def mutate(self):
        """
        Selects a synapse to mutate at random.
        :return: None
        """

        choice = random.randint(0, 1)
        if choice == 0:
            rows, cols = self.h_synapses.shape
            src = random.randint(0, cols - 1)
            dest = random.randint(0, rows - 1)
            # modify h_synapses
            curr_val = self.h_synapses[dest, src]
            self.h_synapses[dest, src] = random.gauss(curr_val, curr_val)
        else:
            # modify m_synapses
            rows, cols = self.m_synapses.shape
            src = random.randint(0, cols - 1)
            dest = random.randint(0, rows - 1)
            # modify m_synapses
            curr_val = self.m_synapses[dest, src]
            self.m_synapses[dest, src] = random.gauss(curr_val, curr_val)

    def compute_initial_state(self, cmds):
        """
        Calculates the birth state of given robot for the given command by calculating the internal states
         of the hidden neurons that should be sent to pyrosim.
        command.
        :param cmds: 2d matrix. each row represents a neuron. the columns represent the connections to each
            other neuron. ex. result[0,0] represents the connection from neuron 0 (col 0) to itself (row 0).
        :return: A tuple where the first element is a list of the last activation state of the hidden neurons
                and the second element is the current adtivation state of the hidden neurons.
        :rtype: (numpy.ndarray, numpy.ndarray)
        """
        if hasattr(self, "tau"):
            tau = self.tau
        else:
            tau = 1.0  # matches default of pyrosim -- quadrupeds do not override this yet
        alpha = 1.0  # matches default of pyrosim -- quadrupeds do not override this yet

        # create 2d Matrix of synapses, and activation states
        synapses = self.get_hidden_neuron_synapses()

        current_activations = numpy.zeros(self.num_hidden_neurons)
        last_activations = numpy.zeros(self.num_hidden_neurons)
        inputs = numpy.zeros(
            self.num_hidden_neurons + 1)  # 5 hidden neurons + 1 auditory neuron for input to 5 hidden neurons.

        for cmd in cmds:
            for val in cmd:
                last_activations = current_activations  # keep track of last activation states
                inputs[0:self.num_hidden_neurons] = current_activations  # update the new inputs
                inputs[-1:] = val
                # print(synapses)
                # print(inputs)
                activation = numpy.dot(synapses, inputs)
                current_activations = numpy.tanh(alpha * activation + tau * last_activations)

        return last_activations, current_activations

    def get_hidden_neuron_synapses(self):
        """
        :return: 2d matrix. each row represents a neuron. the columns represent the connections to each other neuron.
                ex. result[0,0] represents the connection from neuron 0 (col 0) to itself (row 0).
        """
        return_val = self.h_synapses[:, 0:self.num_hidden_neurons + 1]

        return return_val

    def preform_prenatal_development(self, encoding):
        """
        Preforms prenatal development for a given robot by updating the brain of the robot to store the initial internal
        states of the hidden neurons to allow robots to enter the pyrosim simulation engine having already heard the
        command.
        :param encoding: An iterable of iterables (Ex. a 2d array or matrix). Each object in returned by the first
        iterable represents a vectorized word in a command that may contain multiple words.
        Each vectorized word must be iterable as well.
        :return: None
        """
        # Each object returned by the encoding also needs to be iterable.
        # Ex. A 2d array or matrix. This is not checked here.
        assert hasattr(encoding, "__iter__"), "encoding must be iterable. Recieved: " + str(type(encoding))

        # compute the initial internal state of the hidden neurons
        initial_state = self.compute_initial_state(encoding)

        # update the internal states of the hidden neurons
        self.set_hidden_neuron_state(initial_state[0], initial_state[1])

    def set_hidden_neuron_state(self, last_values, values):
        for i in range(len(last_values)):
            self.hidden_neurons_state[i] = (last_values[i], values[i])

    def send_to_simulator(self, simulator, command_encoding):
        if not hasattr(self, 'num_hidden_neurons'):
            # print("Old type of robot")
            self.num_hidden_neurons = 5
            self.tau = 0
        self.preform_prenatal_development(command_encoding)
        self.send_body_to_simulator(simulator)
        self.send_synapses_to_simulator(simulator)

    def send_synapses_to_simulator(self, simulator):
        # send synapses connecting TO hidden neurons ( including recurrent hidden neuron connections.)
        # get shape of self.h_synapses.
        # get number of hidden neurons, number of auditory neurons, number of has_sensors neurons.
        # columns come in the above described order. do not send auditory synapses.

        rows, cols = self.h_synapses.shape

        num_h = len(self.hidden_neurons)
        num_a = 1

        for r in range(rows):
            for c in range(0, num_h):
                simulator.send_synapse(self.hidden_neurons[c], self.hidden_neurons[r], self.h_synapses[r, c])
            # skip the auditory synapses. They are stored in the middle for ease of splicing and prenatal development
            for i, c in enumerate(range(num_h + num_a, cols)):
                simulator.send_synapse(self.sensor_neurons[i], self.hidden_neurons[r], self.h_synapses[r, c])

        # send synapses connecting TO motor neurons
        # get shape of self.m_synapses.
        rows, cols = self.m_synapses.shape

        for r in range(rows):
            for c in range(0, num_h):
                simulator.send_synapse(self.hidden_neurons[c], self.motor_neurons[r], self.m_synapses[r, c])

    def send_body_to_simulator(self, simulator):
        """
        sends the body of the robot to the simulator
        also sends the neurons of the robot to the simulator. does not send the synapses.
        :param simulator: the pyrosim simulator
        :return: None
        """
        L = 0.1
        R = L/5
        head = simulator.send_sphere(x=0, y=0, z=L, mass=0.5,
                                     radius=L, r=0, g=1, b=1)
        pendulum = simulator.send_cylinder(x=0, y=0, z=L, r1=1, r2=0, r3=0,
                                         length=L, radius=R/2, r=0, g=1, b=1)
        connector = simulator.send_sphere(x=(L-R)/2, y=0, z=L,
                                     radius=R, r=0, g=1, b=1)
        joints = [None]*2
        joints[0] = simulator.send_hinge_joint(first_body_id=0, second_body_id=1,
                                            n1=0, n2=1, n3=0, x=0, y=0, z=L, lo=-math.pi/2, hi=math.pi/2)

        if self.second_joint:
            joints[1] = simulator.send_hinge_joint(first_body_id=1, second_body_id=2,
                                                n1=-1, n2=0, n3=0, x=0, y=0, z=L, lo=-math.pi/2, hi=math.pi/2)
        else:
            joints[1] = simulator.send_hinge_joint(first_body_id=1, second_body_id=2,
                                                   n1=-1, n2=0, n3=0, x=0, y=0, z=L, lo=0, hi=0)

        if self.num_sensors == 1:
            self.sensor_neurons[0] = simulator.send_proprioceptive_sensor(joint_id=0)
        position_sensor = simulator.send_position_sensor()

        for i in range(2):
            self.motor_neurons[i] = simulator.send_motor_neuron(joint_id=joints[i], tau=0.3)
        for i in range(self.num_hidden_neurons):
            self.hidden_neurons[i] = simulator.send_hidden_neuron(last_value=self.hidden_neurons_state[i][0],
                                                                  value=self.hidden_neurons_state[i][1])
