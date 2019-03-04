# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import math
import numpy as np
import random
class InputError(Exception):
    pass
NULL_GEM = -1
GEM_NUM_FOR_MATCH = 3
SEED = 7
class GameState:
    def __init__(self,rows,cols,gem_type_count,turn_limit,rand_state=random.getstate()):
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
        self.turn_num = 0
            #each swap increments turn_num
        self.turn_limit = turn_limit
            #number of turns before determined to be "done"
        self.board = np.zeros((rows,cols),dtype=int)
        self.randomize_board()     
            
    def advance_state(self,p1,p2):
        self._swap(p1,p2)
        remove_set = self.get_matches()
        reward = 0
        if len(remove_set) == 0:
            self._swap(p1,p2)
            reward = -10
        else:
            reward = self._process_matches() + self._evaluate_board()
        self.turn_num += 1

        next_state = self.board
        done = self.turn_num >= self.turn_limit
        return self.board, reward, done

    def _swap(self,p1,p2):
        if not self.valid(p1,p2):
            raise InputError(self.board,p1,p2)
        t = self.board[p1[0]][p1[1]]
        self.board[p1[0]][p1[1]] = self.board[p2[0]][p2[1]]
        self.board[p2[0]][p2[1]] = t

    def print_board(self):
        print(self.board)
        
    def _process_matches(self):
        first_iteration= True
        remove_set = set()
        prev_matches = self.gems_matched
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
            return self.generate_gem()
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
        
    def generate_gem(self):
        '''
        Generate random gem using the random state of this GameState object
        '''
        random.setstate(self.rand_state)
        result = random.randint(0,self.gem_type_count-1)
        self.rand_state = random.getstate()
        return result


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
    g = GameState(8,8,7,state)
    #f = GameState(6,6,4,state)
    
    text = ""
    while(text != 'quit'):
        print("\nTurn #"+str(g.turn_num))

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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
