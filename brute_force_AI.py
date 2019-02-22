# -*- coding: utf-8 -*-
from game import GameState
import copy
def get_best_pair(gs):
    max_points = 0
    best_pair = ((0,0),(0,1))
    explored_set = set()
    gs_copy = copy.deepcopy(gs)
    for i in range(0,gs.rows):
        for j in range(0,gs.cols):
            
            p1 = (i,j)
            for p2 in get_adjacent_points(p1,gs.rows,gs.cols):
                if (p1,p2) in explored_set:
                    continue
                explored_set.add((p1,p2))
                gs_copy = copy.deepcopy(gs)
                gs_copy.advance_state(p1,p2)
                if(gs_copy.score > max_points):
                    max_points = gs_copy.score
                    best_pair = (p1,p2)
    return best_pair[0],best_pair[1]

    
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
    
    param = input("Enter turn limit: ")
    turn_limit = int(param)
    
    g = GameState(8,8,7)

    while(g.turn_num < turn_limit):
        #print("\nTurn #"+str(g.turn_num))
        #g.print_board()
        #print("Gems Matched: " + str(g.gems_matched))
        p1,p2 = get_best_pair(g)
        g.advance_state(p1,p2)
    print("Loop exit")
    print("Score was " + str(g.score))