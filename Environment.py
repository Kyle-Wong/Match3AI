import math
import numpy as np
import random
from pybrain.rl.environments.environment import Environment
class InputError(Exception):
    pass
NULL_GEM = -1
GEM_NUM_FOR_MATCH = 3
SEED = 7
class Match3Environment(Environment):
    def __init__(self,rows,cols,gem_type_count,rand_state=random.getstate()):
        self.rand_state = rand_state
            #saved state of random number generator
        self.gem_type_count = gem_type_count
            #domain of gems is [0:gem_type_count]
        self.rows = rows
        self.cols = cols
        self.score = 0
            #score in current GameState (subject to multipliers/combos)
        self.gems_matched = 0
            #Raw number of gems cleared through matching
        self.moves_taken = 0
            #For debug purposes, number of moves taken do not impact game
        self.board = np.zeros((rows,cols),dtype=int)
        self.gem_count = np.arange(0,gem_type_count,dtype=int)
        self.randomize_board()
        self.no_moves_left = False
        self.current_reward = 0
        self.actions = get_all_pairs(self.rows,self.cols)
            #updated when advance_state is called
            
    def advance_state(self,p1,p2):
        #Swap gems
        self._swap(p1,p2)

        #Check if matches were made
        remove_set = self.get_matches()

        #Swap back if no matches
        if len(remove_set) == 0:
            self._swap(p1,p2)
            reward = -10
        else:
            reward = self._process_matches()

        #Check if there are any valid moves left
        self.calculate_if_moves_left()
        done = self.no_moves_left

        #Update reward of action just taken
        self.current_reward = reward
        self.print_board()
        if done:
            self.reset()
        return self.board, reward, done

    def move_is_valid(self, p1, p2):
        self._swap(p1,p2)
        remove_set = self.get_matches()
        self._swap(p1, p2)
        return len(remove_set) > 0

    def get_valid_moves(self):
        result = []
        moves = get_all_pairs(self.rows, self.cols)
        for i in range(len(moves)):
            if self.move_is_valid(moves[i][0], moves[i][1]):
                result.append(i)
        return result

    def _swap(self,p1,p2):
        if not self.valid(p1,p2):
            raise InputError(self.board,p1,p2)
        t = self.board[p1[0]][p1[1]]
        self.board[p1[0]][p1[1]] = self.board[p2[0]][p2[1]]
        self.board[p2[0]][p2[1]] = t

    def print_board(self):
        print(self.board)

    def calculate_if_moves_left(self):
        moves = get_all_pairs(self.rows, self.cols)
        for i in range(len(moves)):
            if self.move_is_valid(moves[i][0], moves[i][1]):
                self.no_moves_left = False
                return True
        self.no_moves_left = True
        return False

    def moves_left(self):
        return not self.no_moves_left

    def _process_matches(self):
        first_iteration = True
        prev_matches = self.gems_matched
        remove_set = set()
        while(first_iteration or len(remove_set) > 0):
            remove_set = self.get_matches()
            if len(remove_set) > 0:
                self.remove_null_gems(remove_set)
            self.gems_matched += len(remove_set)
            self.score += len(remove_set)
            first_iteration = False

        return self.gems_matched - prev_matches
        
    def get_matches(self):
        remove_set = set()
        self._get_vertical_matches(remove_set)
        self._get_horizontal_matches(remove_set)
        return remove_set
    def remove_null_gems(self,remove_set):
        for i,j in remove_set:
            self.gem_count[self.board[i,j]] -= 1
            self.board[i,j] = NULL_GEM
        self._settle_board()
    def _get_vertical_matches(self,remove_set):
        '''
        Get all vertical matches of 3 or more and add the points to remove_set
        '''
        for i in range(0,self.cols):
            self._get_matches_in_array('col',self.board[:,i],remove_set,col=i)
            
    def _get_horizontal_matches(self,remove_set):
        '''
        Get all horizontal matches of 3 or more and add the points to remove_set
        '''
        for i in range(0,self.rows):
            self._get_matches_in_array('row',self.board[i],remove_set,row=i)
            
    def _get_matches_in_array(self,shape,array,remove_set,row=-1,col=-1):
        '''
        Given a 1 dimensional array, add all gems in groups of 3 or more to the remove_set
        '''
        start_index = 0
        gem_count = 1
        gem_type = -1
        for i,gem in enumerate(array):
            if gem == gem_type:
                gem_count += 1
            else:
                if(gem_count >= GEM_NUM_FOR_MATCH):
                    self._add_to_set(start_index,i,shape,array,remove_set,row,col)
                gem_type = gem
                gem_count = 1
                start_index = i
        if(gem_count >= GEM_NUM_FOR_MATCH):
            self._add_to_set(start_index,len(array),shape,array,remove_set,row,col)
        
    def _add_to_set(self,start,end,shape,array,remove_set,row=-1,col=-1):
        '''
        Add all points between start and end to the remove_set.
        Row/Col index must be specified for adding points.
        '''
        for i in range(start,end):
            if(shape == 'row'):
                remove_set.add((row,i))
            elif(shape == 'col'):
                remove_set.add((i,col))
                
    def in_bounds(self,p):
        return p[0] >= 0 and p[0] < self.rows and p[1] >= 0 and p[1] < self.cols
    
    def valid(self,p1,p2):
        '''
        A move is valid if the two points are 1 space away from each other
        and both points are in the board.
        Does not check if swap will make a match.
        '''
        dist = math.fabs(p1[0]-p2[0])+math.fabs(p1[1]-p2[1])
        return self.in_bounds(p1) and self.in_bounds(p2) and dist == 1
    
    def _settle_board(self):
        '''
        Scan left to right, bottom to top.  For each gem, if it is NULL, swap with the
        first non-NULL gem above it, or a randomly generated gem if the top of the board is reached
        '''
        for i in range(self.rows-1,-1,-1):
            for j in range(0,self.cols):
                if(self.board[i,j] == NULL_GEM):
                    self.board[i,j] = self._copy_above((i,j))
                    
    def _copy_above(self,p):
        #Remember p is (row,col),not (x,y)
        if(p[0] <= 0):
            #top row
            #find lowest null gem
            pb = p
            while pb[0]+1 < self.rows-1 and self.board[pb[0]+1,pb[1]] == NULL_GEM:
                pb = (pb[0]+1, pb[1])
            #exclude neighbors to prevent a chance match
            exclude = [self.board[pb[0]+1,pb[1]]]
            if pb[1] > 0:
                exclude.append(self.board[pb[0],pb[1]-1])
            if pb[1] < self.cols - 1:
                exclude.append(self.board[pb[0],pb[1]+1])
            return self.generate_gem(exclude)
        elif(self.board[p[0]-1,p[1]] == NULL_GEM):
            #gem above is NULL_GEM,
            return self._copy_above((p[0]-1,p[1]))
        else:
            #gem above is not NULL_GEM: make it NULL_GEM and return its type
            temp = self.board[p[0]-1,p[1]]
            self.board[p[0]-1,p[1]] = NULL_GEM
            return temp

    def _evaluate_board(self):
        '''
        Return a utility measurement for the current board
        Currently, returns the number of pairs of equal, adjacent gems
        '''
        result = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if i < self.rows - 1 and self.board[i,j] == self.board[i+1,j]:
                    result += 1
                if j < self.cols - 1 and self.board[i,j] == self.board[i,j+1]:
                    result += 1

        return result

    
    
    def randomize_board(self):
        for i in range(0,self.rows):
            for j in range(0,self.cols):
                self.board[i,j] = self.generate_gem()
        self._process_matches()
        self.gems_matched = 0
        self.score = 0

    def generate_gem(self, exclude=[]):
        '''
        Generate random gem using the random state of this GameState object
        '''
        random.setstate(self.rand_state)
        result = random.randint(0,self.gem_type_count-1)
        while result in exclude:
            result = random.randint(0,self.gem_type_count-1)
        self.gem_count[result] += 1

        self.rand_state = random.getstate()
        return result
    def freq_board(self):
        freq_board = np.zeros((self.rows,self.cols),dtype=int)
        gems = []
        for i in range(0,self.gem_type_count):
            #Tuple (frequency, distance from origin (top left), gem type)
            gems.append((self.gem_count[i],self._distance_from_origin(i),i))
        #Sort by frequency first, then distance-from-origin 
        #(doesn't matter if same-frequency gems are ordered from furthest-from-origin or closest-to-origin)
        gems.sort(reverse = True)
        #print(gems)
        gem_map = {}
        for i in range(0, len(gems)):
            #from most frequent to least frequent, assign 0 to gem_type_count
            gem_map[gems[i][2]] = i
        for i in range(0,self.rows):
            for j in range(0,self.cols):
                freq_board[i,j] = gem_map[self.board[i,j]]
        return freq_board

    def _distance_from_origin(self,gem_type):
        for i in range(0,self.rows):
            for j in range(0,self.cols):
                if self.board[i,j] == gem_type:
                    return i*self.cols+j
        return self.rows*self.cols

    def _to_int(self,board_mask):
        '''
        Convert from board of 1's and 0's into
        single integer
        '''        
        return int(''.join(board_mask.flatten()),2)

    def _get_mask(self,gem_type):
        '''
        return a board where gems that are gem_type
        are 1 and others are 0
        '''
        mask = np.zeros((self.rows,self.cols),dtype=str)
        for i in range(0,self.rows):
            for j in range(0,self.cols):
                mask[i,j] = '1' if (self.board[i,j] == gem_type) else '0'
        return [self._to_int(mask)]
    
    def _get_obs(self):
        '''
        Return an observation of the board
        Returns an array of length Gem_type_count
        of integers representing each gem type's
        places on the board
        '''
        masks = []
        for i in range(0,self.gem_type_count):
            masks.append(self._get_mask(i))
        return masks

    '''
    -----------Pybrain environment interface ------------------------
    '''

    def getSensors(self):
        '''
        The currently visible state of the world.
        Returns an array of gem masks
        '''
        return self._get_obs()

    def performAction(self,action):
        '''
        Perform an action on the world that changes it's internal state
        'action' is an array with a single int index representing an swap
        '''
        print(action[0])
        p1,p2 = self.actions[int(action[0])]
        self.advance_state(p1,p2)

    def reset(self):
        print("BOARD RESET")
        self.gem_count = np.arange(0,self.gem_type_count,dtype=int)
        self.no_moves_left = False
        self.randomize_board()
        return self.board

    def currentReward(self):
        return self.current_reward

    


def get_all_pairs(rows,cols):
    result_set = set()
    for i in range(0,rows):
        for j in range(0,cols):
            get_adjacent_points((i,j),rows,cols,result_set)
    result = list(result_set)
    result.sort(key = lambda x : x[1])
    result.sort(key = lambda x : x[0])
    return result
    
    
def get_adjacent_points(p1,rows,cols,result_set):
    if p1[0]-1 >= 0:
        swap = (p1[0]-1,p1[1])
        if not swap_in_set((p1,swap),result_set):
            result_set.add((p1,swap))
    if p1[0]+1 < rows:
        swap = (p1[0]+1,p1[1])
        if not swap_in_set((p1,swap),result_set):
            result_set.add((p1,swap))
    if p1[1]-1 >= 0:
        swap = (p1[0],p1[1]-1)
        if not swap_in_set((p1,swap),result_set):
            result_set.add((p1,swap))
    if p1[1]+1 < cols:
        swap = (p1[0],p1[1]+1)
        if not swap_in_set((p1,swap),result_set):
            result_set.add((p1,swap))
def swap_in_set(swap,swap_set):
    return (swap[0],swap[1]) in swap_set or (swap[1],swap[0]) in swap_set

if __name__ == "__main__":
    
    random.seed(SEED)
    state = random.getstate()
    g = Match3Environment(8,8,7,state)
    text = ""
    while(text != 'quit'):
        print("\nTurn #"+str(g.moves_taken))

        g.print_board()
        print("Gems Matched: " + str(g.gems_matched))
        
        text = input("Enter quit or point1 (r,c): ")
        if text == 'quit':
            break
        p1 = text.split(',')
        text = input("Enter quit or point2 (r,c): ")
        if text == 'quit':
            break
        p2 = text.split(',')
        
        g.advance_state((int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1])))
    print("Loop exit")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
