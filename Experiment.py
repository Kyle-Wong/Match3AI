from pybrain.rl.experiments import Experiment

class Match3Experiment(Experiment):
    #An experiment matches up a task with an agent 
    # and handles their interactions.

    '''
    ALREADY IMPLMENTED IN PARENT
    THIS IS JUST FOR REFERENCE
    
    def doInteractions(self, number = 1):
        """ The default implementation directly maps the methods of the agent and the task.
            Returns the number of interactions done.
        """
        for _ in range(number):
            self._oneInteraction()
        return self.stepid
    '''
    def _oneInteraction(self):
        """ Give the observation to the agent, takes its resulting action and returns
            it to the task. Then gives the reward to the agent again and returns it.
        """
        self.stepid += 1
        self.agent.integrateObservation(self.task.getObservation())
        self.task.performAction((self.agent.getAction(), self.agent.lastobs)) #Bundle board mask with action taken
        reward = self.task.getReward()
        self.agent.giveReward(reward)
        return reward
