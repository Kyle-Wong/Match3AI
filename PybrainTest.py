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
import pickle
import os.path
import numpy as np


from Agent import Match3Agent
from Environment import Match3Environment
from Controller import Match3ActionValueTable
from Experiment import Match3Experiment
from Task import Match3Task

ROWS = 4
COLS = 4
GEM_TYPE_COUNT = 7
SEED = 7
BATCH_SIZE = 100
OUTPUTFILE = "TrainedAIParams"
SAVE = True
LOAD = True

def load_params(file_name,action_value_table):
    current_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_path,file_name)
    if os.path.getsize(file_path) <= 0:
        return

    file = open(file_path,'rb')
    controller._setParameters(pickle.load(file))
    print("Loading: " + str(controller.params))
    
def save_params(file_name,action_value_table):
    current_path = os.path.dirname(os.path.realpath(__file__))
    file = open(os.path.join(current_path,file_name),'wb')
    pickle.dump(controller.params,file)
    print("Saving: " + str(controller.params))

def average_splice(a, n):
    '''
    Return the average values of n splices over collection a
    '''
    result = []
    splice = len(a) / n
    for i in range(n):
        result.append(np.average(a[int(splice*i):int(splice*(i+1))]))
    return np.arange(0, len(a), splice), result

def graph_results(data):
    '''
    Given results of each trial, record human-understandable results
    '''
    x, y = average_splice(data, 20)
    plt.plot(x, y)
    plt.title("Smoothed Learning Curve")
    plt.xlabel("Move #")
    plt.ylabel("Reward")
    plt.show()
    
    y = data[:100]
    plt.plot(np.arange(100), y)
    plt.title("First 100 Move Learning")
    plt.xlabel("Move #")
    plt.ylabel("Reward")
    plt.show()
    
    y = data[-100:]
    plt.plot(np.arange(100), y)
    plt.title("Last 100 Move Learning")
    plt.xlabel("Move #")
    plt.ylabel("Reward")
    plt.show()

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
    num_episodes = num_states * num_actions
    i = 0
    
    try:
        if LOAD:
            load_params(OUTPUTFILE,controller)
                
        while i < num_episodes:
            #AI has no memory of past states
            #learner resets in agent.reset()
            experiment.doInteractions(1)
            #agent.reset()
            i += 1
            agent.learn()
            print(np.shape(where(learner.module.params==1))[1], "unexplored")
            #if i % BATCH_SIZE == 0:
                #agent.learn()
                #agent.history.clear()
                #print(np.shape(where(learner.module.params==1))[1], "unexplored")
                #print(float(i) / num_episodes)
        graph_results(environment.reward_store)
        
    except KeyboardInterrupt:
        pass
    
    if SAVE:
        save_params(OUTPUTFILE,controller)

