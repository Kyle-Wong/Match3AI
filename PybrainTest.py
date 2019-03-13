from scipy import *
import sys, time
from pybrain.rl.environments.mazes import Maze, MDPMazeTask
from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, SARSA
from pybrain.rl.experiments import Experiment
from pybrain.rl.environments import Task
import pylab
import random
import matplotlib.pyplot as plt

from Agent import Match3Agent
from Environment import Match3Environment
from Controller import Match3ActionValueTable
from Experiment import Match3Experiment
from Task import Match3Task

ROWS = 4
COLS = 4
GEM_TYPE_COUNT = 7
SEED = 7
if __name__ == "__main__":
    
    #Instantiate the environment with numInputs and numOutputs
    random.seed(SEED)
    rand_state = random.getstate()
    num_states = 2**(ROWS*COLS)
    num_actions = COLS * (ROWS - 1) + ROWS * (COLS - 1)
    environment = Match3Environment(ROWS,COLS,GEM_TYPE_COUNT,rand_state)
    controller = Match3ActionValueTable(num_states, num_actions)
    controller.initialize(1.)

    learner = Q()
    agent = Match3Agent(controller, learner)
    task = Match3Task(environment)

    experiment = Match3Experiment(task, agent)
    i = 0
    while i < 100:
        #AI has no memory of past states
        #learner resets in agent.reset()
        experiment.doInteractions(1)
        agent.learn()
        agent.reset()
        i += 1
    plt.plot(environment.reward_store)
    plt.show()
