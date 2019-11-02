import math
import random
import numpy


class Quadruped(object):
    def __init__(self, sensors=True, num_hidden_neurons=5):
        self.has_sensors = sensors
        self.num_hidden_neurons = num_hidden_neurons

        # synapses. Initialize to random weights on a normal distribution with a mean of and a standard deviation of 1.
        if self.has_sensors:
            self.h_synapses = numpy.random.normal(0, 1, (self.num_hidden_neurons, 5+self.num_hidden_neurons))  # synapses connecting TO hidden neurons
            self.m_synapses = numpy.random.normal(0, 1, (8, self.num_hidden_neurons))  # synapses connecting to motor neurons
            self.num_sensors = 4
        else:
            self.h_synapses = numpy.random.normal(0, 1, (self.num_hidden_neurons, self.num_hidden_neurons+1))
            self.m_synapses = numpy.random.normal(0, 1, (8, self.num_hidden_neurons))  # synapses connecting to motor neurons
            self.num_sensors = 0

        self.sensor_neurons = {}  # key is neuron number index at 0, value is pyrosim neuron id

        self.hidden_neurons_state = [(0, 0)] * self.num_hidden_neurons  # array of tuples storing (last_value, value) of each hidden neuron.
        self.hidden_neurons = {}  # key is neuron number index at 0, value is pyrosim neuron id

        self.motor_neurons = {}  # key is neuron number index at 0, value is pyrosim neuron id

    def __str__(self):
        return str((self.h_synapses, self.m_synapses))
    
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
            src = random.randint(0, cols-1)
            dest = random.randint(0, rows-1)
            # modify h_synapses
            curr_val = self.h_synapses[dest, src]
            self.h_synapses[dest, src] = random.gauss(curr_val, curr_val)
        else:
            # modify m_synapses
            rows, cols = self.m_synapses.shape
            src = random.randint(0, cols-1)
            dest = random.randint(0, rows-1)
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
            tau = 1.0 # matches default of pyrosim -- quadrupeds do not override this yet
        alpha = 1.0 # matches default of pyrosim -- quadrupeds do not override this yet

        # create 2d Matrix of synapses, and activation states
        synapses = self.get_hidden_neuron_synapses()

        current_activations = numpy.zeros(self.num_hidden_neurons)
        last_activations = numpy.zeros(self.num_hidden_neurons)
        inputs = numpy.zeros(self.num_hidden_neurons+1)                 # 5 hidden neurons + 1 auditory neuron for input to 5 hidden neurons.

        for cmd in cmds:
            for val in cmd:
                last_activations = current_activations      # keep track of last activation states
                inputs[0:self.num_hidden_neurons] = current_activations           # update the new inputs
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
        return_val = self.h_synapses[:, 0:self.num_hidden_neurons+1]

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
        if not hasattr(self,'num_hidden_neurons'):
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
            for i, c in enumerate(range(num_h+num_a, cols)):
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
        length = 0.1
        radius = length/5

        # id arrays
        thighs = [0] * 4
        shins = [0] * 4
        hips = [0] * 4
        knees = [0] * 4
        foot_sensors = [0] * 4

        delta = float(math.pi) / 2.0
        main_body = simulator.send_box(x=0, y=0, z=length + radius,
                                       length=length, width=length,
                                       height=radius * 2.0, mass=1,
                                       r=0.8, g=0, b=0.8)
        # quadruped is a box with one leg on each side
        # each leg consists thigh and shin cylinders
        # with hip and knee joints
        # each shin/foot then has a touch has_sensors
        for i in range(4):
            theta = delta * i
            x_pos = math.cos(theta) * length
            y_pos = math.sin(theta) * length

            thighs[i] = simulator.send_cylinder(x=x_pos, y=y_pos, z=length + radius,
                                                r1=x_pos, r2=y_pos, r3=0,
                                                length=length, radius=radius, capped=True,
                                                r=0.5, g=0.5, b=1)

            hips[i] = simulator.send_hinge_joint(main_body, thighs[i],
                                                 x=x_pos / 2.0, y=y_pos / 2.0, z=length + radius,
                                                 n1=-y_pos, n2=x_pos, n3=0,
                                                 lo=-math.pi / 4.0, hi=math.pi / 4.0,
                                                 speed=1.0)

            x_pos2 = math.cos(theta) * 1.5 * length
            y_pos2 = math.sin(theta) * 1.5 * length

            shins[i] = simulator.send_cylinder(x=x_pos2, y=y_pos2, z=length / 2 + radius,
                                               r1=0, r2=0, r3=1,
                                               length=length, radius=radius,
                                               mass=1., capped=True,
                                               r=0, g=1, b=1)

            knees[i] = simulator.send_hinge_joint(thighs[i], shins[i],
                                                  x=x_pos2, y=y_pos2, z=length + radius,
                                                  n1=-y_pos, n2=x_pos, n3=0,
                                                  lo=-math.pi / 4.0, hi=math.pi / 4.0)
            foot_sensors[i] = simulator.send_touch_sensor(shins[i])

        # Send the position sensor. Used for the fitness function.
        simulator.send_position_sensor(main_body)

        for i in range(4):
            self.motor_neurons[i] = simulator.send_motor_neuron(joint_id=hips[i], tau=0.3)

        for i in range(4):
            self.motor_neurons[i + 4] = simulator.send_motor_neuron(knees[i])

        if self.has_sensors:
            for i in range(4):
                self.sensor_neurons[i] = simulator.send_sensor_neuron(foot_sensors[i])

        for i in range(self.num_hidden_neurons):
            self.hidden_neurons[i] = simulator.send_hidden_neuron(last_value=self.hidden_neurons_state[i][0],
                                                                  value=self.hidden_neurons_state[i][1])
    def get_motion(self, sim_dat):
        return 1

