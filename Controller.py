from pybrain.rl.learners.valuebased.interface import ActionValueTable
from scipy import where
from random import choice

class Match3ActionValueTable(ActionValueTable):
    #ActionValueTable inherts from Table and ActionValueInterface
    #A Table is a Module
    """ A special table that is used for Value Estimation methods
        in Reinforcement Learning. This table is used for value-based
        TD algorithms like Q or SARSA.
    """

    def getMaxAction(self,state):
        #state is environment
        '''
        Return the action with the maximal value for the given state.
        This is a slightly modified form of ActionValueTable.getMaxAction.

        Method inherited from Pybrain.ActionValueTable
        '''
        values = self.params.reshape(self.numRows, self.numColumns)[int(state), :].flatten()
        self.maxvalue = max(values)
        action = where(values == self.maxvalue)[0]
        action = choice(action)
        
        return action

    def getUnexploredAction(self,state,value=1.0,default=[0]):
        #state is environment
        '''
        Return an action with a value equal to the given one
        '''
        values = self.params.reshape(self.numRows, self.numColumns)[int(state), :].flatten()
        action = where(values == value)[0]
        if default[0] in action or len(action) == 0:
            return default[0]
        
        action = choice(action)
        return action

