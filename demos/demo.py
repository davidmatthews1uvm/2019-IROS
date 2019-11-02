import random
import sys
import numpy

sys.path.insert(0, "..")

# import the individual wrapper
from experiments.w2v_robot import W2VRobot

# import the robot morphologies
from experiments.w2v_vecs import *
from experiments.quadruped import Quadruped
from experiments.twig import Twig
from experiments.spherebot import SphereBot
from demos.word2vecDatabase import Word2VecVectorSpace


if __name__ == '__main__':
    assert len(sys.argv) >= 2, "please run as `python job.py <word to send>`"
    db = Word2VecVectorSpace(database_file='w2vVectorSpace-google.db')

    word = sys.argv[1]
    wordVec = None
    try:
        wordVec = db.get_vector(word)
    except KeyError:
        print("Vector not found for given word, %s" % word)
        exit(1)

    else:
        print("Vector found for given word, %s" % word)

    train_cmds = {"forward": [[wordVec]]}

    def get_internal_bot():
        return Quadruped()

    def robot_factory():
        internal_robot = get_internal_bot()
        return W2VRobot(internal_robot, train_cmds, eval_time=500)

    print("Sending robot to simulator...\n")
    robot = W2VRobot(get_internal_bot(), train_cmds, eval_time=500)
    robot.play_blind = False
    robot.compute_work(serial=True)
    print("Done")