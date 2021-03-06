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

'''
Run this module to train the AI
REQUIRES PYBRAIN
'''


ROWS = 4
COLS = 4
GEM_TYPE_COUNT = 7
SEED = 7
OUTPUTFILE = "TrainedAIParams"
SAVE = True
LOAD = True
COMPARE_AGAINST_RANDOM = False
STREAK_HISTOGRAM = False
def join_to_base_path(rel_path):
    try:
        base_path = sys._MEI
    except:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, rel_path)
def load_params(file_name,action_value_table):
    #current_path = os.path.dirname(os.path.realpath(__file__))
    #file_path = os.path.join(current_path,file_name)
    file_path = join_to_base_path(file_name)
    if os.path.getsize(file_path) <= 0:
        return

    file = open(file_path,'rb')
    controller._setParameters(pickle.load(file))
    print("Loading: " + str(controller.params))
    
def save_params(file_name,action_value_table):
    #current_path = os.path.dirname(os.path.realpath(__file__))
    file_path = join_to_base_path(file_name)
    file = open(file_path,'wb')
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

def graph_splice_results(data, title, xlabel, ylabel):
    '''
    Given data and labels, turn into a spliced graph
    '''
    x, y = average_splice(data, 20)
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def graph_results(data, title, xlabel, ylabel):
    '''
    Given data and labels, turn into a detailed graph
    '''
    plt.plot(np.arange(len(data)), data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def compare_against_random(turn_limit,rand_state,environment_score_store):
    g = Match3Environment(4,4,7,rand_state)
    i = 0
    while i < turn_limit:
        action = random.randint(0,23)
        g.performAction([(action)])
        i+= 1

    plt.plot(np.arange(len(g.score_store)), g.score_store,'r-',label="Random AI")
    plt.plot(np.arange(len(environment_score_store)), environment_score_store,'b-',label="Pybrain AI")
    plt.legend()
    plt.title("Score over Moves")
    plt.xlabel("Move #")
    plt.ylabel("Total Score")
    plt.show()

def plot_histogram(data, title, xlabel, ylabel):
    plt.hist(data,bins=[0,1,2,3,4,5,6,7,8,9,10,11,12])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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
    num_episodes = 1000
    i = 0
    
    try:
        if LOAD:
            load_params(OUTPUTFILE,controller)
                
        while i < num_episodes:
            experiment.doInteractions(1)
            i += 1
            agent.learn()
        graph_splice_results(environment.reward_store, "Relative Reward per Move", "Move #", "Reward")
        graph_results(environment.score_store, "Score over Moves", "Move #", "Total Score")
        
    except KeyboardInterrupt:
        pass
    if SAVE:
        save_params(OUTPUTFILE,controller)
    print("GOOD-MOVES PERCENT: " + str(environment.good_move_counter/environment.moves_taken*100) + "%")
    if COMPARE_AGAINST_RANDOM:
        compare_against_random(num_episodes,rand_state,environment.score_store)
    if STREAK_HISTOGRAM:
        plot_histogram(environment.streak_store,"Scoring-Action Streak Length", "Streak Length", "Actions taken")
    


    


