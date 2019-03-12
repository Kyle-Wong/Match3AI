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
        '''
        values = self.params.reshape(self.numRows, self.numColumns)[int(state), :].flatten()
        self.maxvalue = max(values)
        action = where(values == self.maxvalue)[0]
        action = choice(action)
        
        return action


