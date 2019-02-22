# -*- coding: utf-8 -*-
from game import GameState
import random

def get_random_pair(rows,cols,ai_state):
    random.setstate(ai_state)
    p1 = (random.randint(0,rows-1),random.randint(0,cols-1))
    neighbors = get_adjacent_points(p1,rows,cols)
    p2 = neighbors[random.randint(0,len(neighbors)-1)]   
    ai_state = random.getstate()
    return p1,p2,ai_state

def get_adjacent_points(p1,rows,cols):
    result = []
    if p1[0]-1 >= 0:
        result.append((p1[0]-1,p1[1]))
    if p1[0]+1 < rows:
        result.append((p1[0]+1,p1[1]))
    if p1[1]-1 >= 0:
        result.append((p1[0],p1[1]-1))
    if p1[1]+1 < cols:
        result.append((p1[0],p1[1]+1))
    return result

if __name__ == "__main__":
    param = input("Enter game seed: ")
    random.seed(int(param))
    state = random.getstate()
    param = input("Enter AI seed: ")
    random.seed(int(param))
    ai_state = random.getstate()
    param = input("Enter turn limit: ")
    turn_limit = int(param)
    
    g = GameState(8,8,7,state)

    while(g.turn_num < turn_limit):
        '''
        print("\nTurn #"+str(g.turn_num))
        g.print_board()
        print("Gems Matched: " + str(g.gems_matched))
        '''
        p1,p2,ai_state = get_random_pair(g.rows,g.cols,ai_state)
        g.advance_state(p1,p2)
    print("Loop exit")
    print("Score was " + str(g.score))
