from pybrain.rl.agents.logging import LoggingAgent
from pybrain.rl.agents.learning import LearningAgent
from scipy import where
from random import choice
from numpy import array
class Match3Agent(LearningAgent):

    #bool self.learning -> inherited from LearningAgent.
    #Returns true if agent is currently learning from experience

    
    #Logging Agent:
    """ This agent stores actions, states, and rewards encountered during
        interaction with an environment in a ReinforcementDataSet (which is
        a variation of SequentialDataSet).
        The stored history can be used for learning and is erased by resetting
        the agent. It also makes sure that integrateObservation, getAction and
        giveReward are called in exactly that order.
    """
    #Learning Agent:
    #LearningAgent inherits from LoggingAgent inherits from Agent
    """ LearningAgent has a module, a learner, that modifies the module, and an explorer,
        which perturbs the actions. It can have learning enabled or disabled and can be
        used continuously or with episodes.
    """
    

    def Match3Agent(self,module,learner=None):
        super(self,module,learner)
        self.actionhistory = []

    def getAction(self):
        '''
        Activate the module with the last observation, 
        add the exploration from the explorer object 
        and store the result as last action.
        '''
        actions = []
            #Each value in actions is the best action for each gem mask
        values = []
            #Value of the corresponding action of each gem mask
        for mask in self.lastobs:
            actions.append(self.module.activate(mask))
            values.append(self.module.maxvalue)
        bestactionindex = getMaxAction(actions,values)
        self.lastaction = actions[bestactionindex]
        cachedaction = self.lastaction
        self.lastobs = self.lastobs[bestactionindex]

        if self.learning:
            self.lastaction = self.learner.explore(self.lastobs, self.lastaction)
            #If random choice, choose random non-explored choice
            if cachedaction != self.lastaction:
                cachedaction = self.lastaction
                self.lastaction = array([self.module.getUnexploredAction(self.lastobs[0], default=self.lastaction)])
                
        #if cachedaction == self.lastaction:
            #print(self.lastaction, ":", values[bestactionindex])
        
        return self.lastaction
    
    def giveReward(self,r):
        """Step 3: store observation, action and 
        reward in the history dataset. """
        self.lastreward = r
        if self.history.getLength() >= 2:
            self.removeOldestHistory(1)
        self.history.addSample(self.lastobs, self.lastaction, self.lastreward)

    def removeOldestHistory(self, n=1):
        """
        Remove the oldest n entries from the history
        """
        assert n < self.history.getLength()
        newHistory = self.history.getSequence(0)
        self.history.clear()
        for i in range(n, len(newHistory[0])):
            self.history.addSample(newHistory[0][i], newHistory[1][i], newHistory[2][i])
                          

def getMaxAction(actions,values):
    maxvalue = max(values)
    maxactions = []
    for i in range(0,len(actions)):
        if values[i] == maxvalue:
            maxactions.append(i)
        
    return choice(maxactions)

