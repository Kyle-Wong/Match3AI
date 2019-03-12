from pybrain.rl.environments.task import Task

class Match3Task(Task):

    def getObservation(self):
        '''
        A filtered mapping to getSample of the underlying environment.
        '''
        return Task.getObservation(self)

    def getReward(self):
        '''
        Compute and return the current reward 
        (i.e. corresponding to the last action performed)
        '''
        return self.env.currentReward()
        
    def performAction(self,action):
        #action is ( (p1.x,p1.y), (p2.x,p2.y) )
        '''
        A filtered mapping towards performAction of 
        the underlying environment.
        '''
        Task.performAction(self,action)
