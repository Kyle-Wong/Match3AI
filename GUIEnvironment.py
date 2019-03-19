from Gem import *
import pygame
import os.path
from enum import Enum
import numpy as np
import math
from pybrain.rl.environments.environment import Environment
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

from Agent import Match3Agent
from Environment import Match3Environment
from Controller import Match3ActionValueTable
from Experiment import Match3Experiment
from Task import Match3Task
import pickle


class State(Enum):
    STANDBY = 1
    CLEARING = 2
    SETTLING = 3

def load_sprites(folder):
    '''
    Loads all sprites in the Match3AI/Sprites folder into the sprites dictionary.
    Key is file name (without extensions) and Value is a Surface of the sprite.
    '''
    current_path = os.path.dirname(os.path.realpath(__file__))
    sprite_path = os.path.join(current_path,folder)
    sprites = {}
    for file in os.listdir(sprite_path):
        sprites[file.split('.')[0]] = pygame.image.load(os.path.join(sprite_path,file))
    print("---------SPRITES----------")
    for s in sprites:
        print(s)
    return sprites
def load_fonts(folder):
    '''
    Loads all fonts in the Match3AI/Fonts folder into the Fonts dictionary.
    Key is file name (without extensions) and Value is the Path to the Font file
    '''
    current_path = os.path.dirname(os.path.realpath(__file__))
    font_path = os.path.join(current_path,folder)
    fonts = {}
    for file in os.listdir(font_path):
        fonts[file.split('.')[0]] = os.path.join(font_path,file)
    print("---------FONTS------------")
    for f in fonts:
        print(f)
    return fonts
def load_sounds(folder):
    '''
    Loads all fonts in the Match3AI/Fonts folder into the Fonts dictionary.
    Key is file name (without extensions) and Value is the pygame sound object
    '''
    current_path = os.path.dirname(os.path.realpath(__file__))
    sound_path = os.path.join(current_path,folder)
    sounds = {}
    for file in os.listdir(sound_path):
        print(file.split('.')[0])
        sounds[file.split('.')[0]] = pygame.mixer.Sound(os.path.join(sound_path,file))
    print("---------SOUNDS-----------")
    for s in sounds:
        print(s)
    return sounds

class MaskInfo():

    def __init__(self, window_index, mirrored, rotations):
        self.index = window_index
        self.mirrored = mirrored
        self.rotations = rotations
    def window_origin(self):
        origin = divmod(self.index,5,dtype = int)
        return origin
    def print_info(self):
        print("Window " + str(self.index))
        print("  w_origin (row,col): " + str(self.window_origin()))
        print("  mirrored          : " + str(self.mirrored))
        print("  rotations         : " + str(self.rotations))

class Match3GUIEnvironment(Environment):

    def __init__(self,rows,cols,gem_type_count,speed,sprites,rand_state = random.getstate()):
        self.sprites = sprites
        self.p_width = 930 #Pixel width
        self.p_height = 600 #Pixel height
        self.speed = 1/speed
        self.gs = Match3Environment(rows,cols,gem_type_count,rand_state) #GameState
        
        self.display = pygame.display.set_mode((self.p_width,self.p_height))
        self.board_rect = pygame.Rect(215,0,500,500) #Game board bounding box
        self.tile_w = self.board_rect.w/self.gs.cols #Width of a board tile
        self.tile_h = self.board_rect.h/self.gs.rows #Height of a board tile
        self.objects = [] #List of objects w/ update() and draw() (currently none)
        self._initialize_board()
        pygame.display.set_caption('Match3')
        self.surface = pygame.Surface((800,600)) #Surface to draw images/shapes on to"
        self.clock = pygame.time.Clock() #Game block
        self.delta_time = 0 #Seconds since last frame
        self.running = True #Quit game if no longer running
        self.state = State.STANDBY #Current game state
        self.prev_move = None #( (p1.x,p1.y), (p2.x,p2.y) ) which is the last action taken
        self.prev_matches = 0 #Gems matched prior to current action
        self.improvement = 0 #self.gems_matched - prev_matches
        self._swap_back = False
        self.total_moves_taken = 0
        self.total_score = 0
        self.mask_dict = {} # {board_as_int : MaskInfo} pairs
        self.windows = self._get_windows(self.gs.board)
        
        self.paused = False
        
    def run(self):
        '''
        Begin main loop, call update and draw each frame
        '''
        
        
        while self.running:
            self.delta_time = self.clock.get_time()/1000/self.speed
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()
        quit()

    def update(self):
        '''
        Update all objects and advance game state
        '''
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                self._handle_key_down(event)
            if event.type == pygame.QUIT:
                self.running = False
        self.update_gems()

        if self.state is State.STANDBY:
            if self.paused:
                return
            if not self._gems_blocking():
                self.AI_train_step()
                
        elif self.state is State.CLEARING:
            if not self._gems_blocking():
                match_set = self.gs.get_matches()
                if self._swap_back:
                    self._swap_gems(self.prev_move[0],self.prev_move[1])
                    self.gs._swap(self.prev_move[0],self.prev_move[1])
                    self.state = State.STANDBY
                    self._swap_back = False

                    return
                for p1 in match_set:
                    self.gems[p1].clear()
                if len(match_set) > 0:
                    sounds["gem_clear"].play()

                self.state = State.SETTLING
        elif self.state is State.SETTLING:
            if not self._gems_blocking():
                self.settle_step()
               
            
                
        
        
    def draw(self):
        '''
        Call the draw method of every object
        '''
        self.surface = pygame.Surface((self.p_width,self.p_height))
        self._draw_background(0,0)
        self._draw_board(self.board_rect.x,self.board_rect.y,self.board_rect.w,self.board_rect.h)
        self._draw_gems()
        self._draw_prev_move()
        self._draw_turns_left(30,30,title_font)
        self._draw_score(self.p_width-150,30,title_font)
        self._draw_credits(10,self.p_height-20,normal_font)
        if self.paused:
            self._draw_pause_overlay(410, 500, title_font)
        self.display.blit(self.surface,(0,0))
    def AI_train_step(self):
        experiment.doInteractions(1)
        if self.total_moves_taken % BATCH_SIZE == 0:
            agent.learn()
            agent.history.clear()
    def update_gems(self):
        for i in range(0,self.gs.rows):
            for j in range(0,self.gs.cols):
                self.gems[i,j].update()

    def advance_state(self,p1,p2,action,mirrored=False,rotations=0):
        '''
        Do a swap and move into SWAP
        Abort if state is not STANDBY
        '''
        '''
        if self.state is not State.STANDBY:
            return
        '''
        self.state = State.CLEARING
        self.prev_matches = self.gs.gems_matched
        self.gs._swap(p1,p2)
        self._swap_gems(p1,p2)
        self.prev_move = (p1,p2)
        matches = self.gs.get_matches()
        window_as_int = action[1][0]
        window = self.gs.int_to_matrix(window_as_int)
        if len(matches) == 0:
            self._swap_back = True
            self.gs.current_reward = -10
            self.gs._reset_streak()
        else:
            self.gs.current_reward = self.gs.get_reward(action)
            self.gs._increment_streak()
            print("REWARD: " + str(self.gs.current_reward) + '\n')

        self.total_moves_taken += 1


    def settle_step(self):
        remove_set = self.gs.get_matches()
        if len(remove_set) > 0:
            self.copy_board_to_gs()
            self._match_gems(remove_set)
            self.copy_board_to_gs()
            match_count = len(remove_set)
            self.gs.gems_matched += match_count
            self.gs.score += match_count
            self.total_score += match_count
            self.state = State.CLEARING
        else:
            self.state = State.STANDBY
            self.improvement = self.gs.gems_matched-self.prev_matches
            if not self.gs.calculate_if_moves_left():
                self.reset()

    def _swap_gems(self,p1,p2):
        '''
        Swap gems in the gems matrix
        and swap their desired grid position
        '''
        temp = self.gems[p1]
        self.gems[p1] = self.gems[p2]
        self.gems[p2] = temp
        self.gems[p1].set_grid(p1)
        self.gems[p2].set_grid(p2)

    def _swap_gems_instant(self,p1,p2):
        '''
        Swap gems in the gems matrix
        and switch their positions instantly
        '''
        temp = self.gems[p1]
        self.gems[p1] = self.gems[p2]
        self.gems[p2] = temp
        self.gems[p1].jump_to_grid(p1)
        self.gems[p2].jump_to_grid(p2)
        self.gems[p1].set_grid(p1)
        self.gems[p2].set_grid(p2)
    def _match_gems(self,remove_set):
        h_stack = np.ones((self.gs.cols),dtype=int)
        h_stack = -1*h_stack
        for point in remove_set:
            self.gems[point].gem_type = -1
            self.gs.board[point] = -1
        for i in range(self.gs.rows-1,-1,-1):
            for j in range(0,self.gs.cols):
                p1 = (i,j)
                if self.gems[p1].gem_type == -1:
                    self.gs.board[p1] = self.gs._copy_above(p1)
                    p2 = self._get_above(p1)
                    if p2 is None:
                        self.gems[p1].jump_to_grid((h_stack[p1[1]],p1[1]))
                        self.gems[p1].gem_type = self.gs.board[p1]
                        h_stack[p1[1]] -= 1
                    else:
                        self._swap_gems(p1,p2)
                        
    def _get_above(self,p1):
        '''
        Get the (r,c) coordinate of the gem above this one (ignoring NULL GEMS)
        '''
        p2 = (p1[0]-1,p1[1])
        if p2[0] < 0:
            return None
        elif self.gems[p2].gem_type == -1:
            return self._get_above(p2)
        else:
            return p2
        
    def _gems_blocking(self):
        '''
        Checks if any gems are currently in motion (false is all animation is finished)
        '''
        for i in range(0,self.gs.rows):
            for j in range(0,self.gs.cols):
                if self.gems[i,j].blocking():
                    return True
        return False
        
    def _handle_key_down(self,event):
        '''
        Handle all key inputs and do appropriate action
        '''
        if event.key == pygame.K_ESCAPE:
            self.running = False
        if event.key == pygame.K_SPACE:
            self.paused = not self.paused
        if event.key == pygame.K_r:
            self.__init__(8,8,7,10,sprites,state)
        

    def _draw_board(self,x,y,width,height):
        '''
        Draw game board background
        '''
        step_w = (int)(width/self.gs.cols)
        step_h = (int)(height/self.gs.rows)
        sub_surface = pygame.Surface((step_w,step_h))
        color = pygame.Color(0,0,0)
        for i in range(0,self.gs.rows):
            for j in range(0,self.gs.cols):
                rect = pygame.Rect(x+j*step_w,y+i*step_h,step_w,step_h)
                sub_surface.fill(pygame.Color(30,30,30))
                if (i+j)%2 == 0:
                    sub_surface.set_alpha(150)
                else:
                    sub_surface.set_alpha(90)
                self.surface.blit(sub_surface,rect)
        pygame.draw.rect(self.surface,pygame.Color(0,0,0)\
                         ,pygame.Rect(x,y,step_w*self.gs.cols,\
                        step_w*self.gs.rows),5)
        
    def _draw_gems(self):
        '''
        Draw all gems
        '''
        for i in range(0,self.gs.rows):
            for j in range(0,self.gs.cols):
                self.gems[i,j].draw(self.surface)
    def _draw_prev_move(self):
        '''
        Highlight last 2 tiles swapped with yellow box
        '''
        if self.prev_move is None:
            return
        p1 = self.prev_move[0]
        p2 = self.prev_move[1]
        rect1 = pygame.Rect(self.board_rect.x+p1[1]*self.tile_w,self.board_rect.y+p1[0]*self.tile_h,\
                            self.tile_w,self.tile_h)
        rect2 = pygame.Rect(self.board_rect.x+p2[1]*self.tile_w,self.board_rect.y+p2[0]*self.tile_h,\
                            self.tile_w,self.tile_h)
        pygame.draw.rect(self.surface,pygame.Color(255,255,0),rect1,3)
        pygame.draw.rect(self.surface,pygame.Color(255,255,0),rect2,3)

    def _draw_background(self,offset_x = 0, offset_y = 0):
        '''
        Draw background image with optional x and y offset (defaults to centered image)
        '''
        x = (self.surface.get_width()-self.sprites['background'].get_width())/2
        y = (self.surface.get_height()-self.sprites['background'].get_height())/2
        x += offset_x
        y += offset_y
        self.surface.blit(self.sprites['background'],(x,y))
    def _draw_turns_left(self,x,y,font):
        text = "Moves Taken:"
        surface = title_font.render(text,True,pygame.Color(0,0,0))
        text_dim = title_font.size(text)
        self._draw_text_outline(text,x,y,text_dim[0],text_dim[1],1.25,font,pygame.Color(255,255,255))
        self.surface.blit(surface,pygame.Rect(x,y,text_dim[0],text_dim[1]))
        
        text = str(self.total_moves_taken)
        surface = title_font.render(text,True,pygame.Color(0,0,0))
        text_dim = title_font.size(text)
        self._draw_text_outline(text,x,y+40,text_dim[0],text_dim[1],1.25,font,pygame.Color(255,255,255))
        self.surface.blit(surface,pygame.Rect(x,y+40,text_dim[0],text_dim[1]))     

    def _draw_score(self,x,y,font):
        text = "Score:"
        surface = font.render(text,True,pygame.Color(0,0,0))
        text_dim = font.size(text)
        self._draw_text_outline(text,x,y,text_dim[0],text_dim[1],1.25,font,pygame.Color(255,255,255))
        self.surface.blit(surface,pygame.Rect(x,y,text_dim[0],text_dim[1]))

        text = str(self.total_score*100)
        surface = font.render(text,True,pygame.Color(0,0,0))
        text_dim = font.size(text)
        self._draw_text_outline(text,x,y+40,text_dim[0],text_dim[1],1.25,font,pygame.Color(255,255,255))
        self.surface.blit(surface,pygame.Rect(x,y+40,text_dim[0],text_dim[1]))
    def _draw_credits(self,x,y,font):
        text = "Gems by limbusdev, Sounds by ViRiX Dreamcore (soundcloud.com/virix)"
        surface = font.render(text,True,pygame.Color(0,0,0))
        text_dim = font.size(text)
        self.surface.blit(surface,pygame.Rect(x,y,text_dim[0],text_dim[1]))
    def _draw_pause_overlay(self,x,y,font):
        overlay = pygame.Surface((self.p_width,self.p_height))
        overlay.fill(pygame.Color(0,0,0))
        overlay.set_alpha(120)
        self.surface.blit(overlay,pygame.Rect(0,0,self.p_width,self.p_height))
        
        text = "PAUSED"
        surface = font.render(text,True,pygame.Color(128,128,0))
        text_dim = font.size(text)
        self._draw_text_outline(text,x,y,text_dim[0],text_dim[1],1.25,font,pygame.Color(255,255,255))
        self.surface.blit(surface,pygame.Rect(x,y,text_dim[0],text_dim[1]))

        text = "Space to Unpause"
        surface = font.render(text,True,pygame.Color(128,128,0))
        text_dim = font.size(text)
        self._draw_text_outline(text,x-65,y+40,text_dim[0],text_dim[1],1.25,font,pygame.Color(255,255,255))
        self.surface.blit(surface,pygame.Rect(x-65,y+40,text_dim[0],text_dim[1]))

        


    def _draw_text_outline(self,text,x,y,width,height,weight,font,color):
        '''
        Basic white outline by drawing the text 8 times with small radial offset
        '''
        offsets = [
            (1,0),
            (.707,.707),
            (0,1),
            (-.707,.707),
            (-1,0),
            (-.707,-.707),
            (0,-1),
            (.707,-.707)
        ] #Unit circle values
        text_surface = font.render(text,True,color)
        for xOffset,yOffset in offsets:
            self.surface.blit(text_surface,pygame.Rect(x+xOffset*weight,y+yOffset*weight,width,height))

    def _initialize_board(self):
        '''
        Spawn gem objects with correct sprite type
        '''
        self.gems = np.empty((self.gs.rows,self.gs.cols),dtype=Gem)
        for i in range(0,self.gs.rows):
            for j in range(0,self.gs.cols):
                self.gems[i,j] = Gem(i,j,self.gs.board[i,j],300,self)
                self.objects.append(self.gems[i,j])

    def copy_board_to_gs(self):
        for i in range(0,self.gs.rows):
            for j in range(0,self.gs.cols):
                self.gs.board[i,j] = self.gems[i,j].gem_type

    
    def _get_windows(self,board):
        '''
        return all 4x4 windows for the game board.
        Use self.windows to avoid repeated calculation
        '''
        result = []
        for r in range(0,self.gs.rows-WINDOW_SIZE+1):
            for c in range(0,self.gs.cols-WINDOW_SIZE+1):
                result.append(board[r:r+4,c:c+4])
        return result
    def get_gem_masks(self,mask_set,gem_type):
        '''
        For a specific gem type, get all combinations of rotations and mirrors as
        masks and return them.
        '''
        board = self.gs.board
        result = []
        for mirrored in range(0,2): #Mirror
            for rotations in range(0,4):  #Rotate 4 times
                for index, window in enumerate(self.windows): #iterate through all windows
                    w = np.fliplr(window) if mirrored==1 else window #apply mirroring
                    w = np.rot90(window,rotations) #apply rotations
                    window_as_int = self.gs._get_mask(w,gem_type)

                    if window_as_int[0] in mask_set: #Skip if mask has already been found (mask_dict can't have duplicate keys)
                        continue

                    mask_set.add(window_as_int[0])   #Do not allow duplicate masks
                    result.append(window_as_int)
                    self.mask_dict[window_as_int[0]] = MaskInfo(index,mirrored==1,rotations) #Associate mask with MaskInfo           
        return result
        
    def get_masks(self):
        '''
        Get all UNIQUE gem masks.  Get masks for each gem, for all 
        combinations of rotations and mirrors.  Store the rotation and
        mirroring data in self.mask_dict translate action for 8x8 board
        '''
        masks = []
        mask_set = set() #contains ints representing 4x4 state
        #self.mask_dict => {int : MaskInfo}
        self.mask_dict = {} #Clear mask_dict between actions
        for i in range(0,self.gs.gem_type_count):
            masks.extend(self.get_gem_masks(mask_set,i))
        return masks
    def _rotate_coord_cw(self,point,rotations):
        '''
        Rotate a coordinate point clockwise on a 4x4 board
        Formula is: (r,c) -> (c,(4-1)-r)
        '''
        for i in range(0,rotations):
            point = (point[1],(WINDOW_SIZE-1)-point[0])
        return point
    def _unmirror_coord(self,point):
        '''
        Translate a coordinate to its mirrored point, mirroring across the vertical axis
        Formula is (r,c) -> (r,(4-1)-c)
        '''
        return (point[0],(WINDOW_SIZE)-1-point[1])
    def _shift_coord(self,point,offset):
        '''
        Do an actual translation from a point by an offset
        '''
        return (point[0]+offset[0],point[1]+offset[1])
    def _translate_action(self,action):
        '''
        action is ([action#{0-23}],[board_as_int]).
        Translate the action number of a 4x4 board to the pair of points
        for a swap on an 8x8 board.
        '''
        
        #Get swap pair RELATIVE TO 4x4 state
        p1,p2 = self.gs.window_actions[int(action[0][0])]
        #Get the board associatec with this action--for lookup for translation values
        board_as_int = action[1][0]

        print(self.gs.window_actions[int(action[0][0])])
        print(self.gs.print_mask(action[1]))
        self.mask_dict[board_as_int].print_info()

        #Get the information about the board
        #so the swap coordinates can be translated
        #to coordinates in the 8x8 board
        mask_info = self.mask_dict[board_as_int]
        offset = mask_info.window_origin()

        #each point must be rotated, mirrored, and then shifted by
        #the window offset, in that order
        p1 = self._rotate_coord_cw(p1,mask_info.rotations)
        p1 = self._unmirror_coord(p1) if mask_info.mirrored else p1
        p1 = self._shift_coord(p1,offset)

        p2 = self._rotate_coord_cw(p2,mask_info.rotations)
        p2 = self._unmirror_coord(p2) if mask_info.mirrored else p2
        p2 = self._shift_coord(p2,offset)

        return (p1,p2)
    '''
    -----------Pybrain environment interface ------------------------
    '''

    def getSensors(self):
        '''
        The currently visible state of the world.
        Returns an array of gem masks
        '''
        return self.get_masks()

    def performAction(self,action):
        '''
        Perform an action on the world that changes it's internal state
        action is ([action#{0-23}],[board_as_int]).
        '''
        p1,p2 = self._translate_action(action)
        print("Translated swap: " + str((p1,p2)))
        self.advance_state(p1,p2,action)

    def reset(self):
        self.gs.reset()
        self._initialize_board()
        self.state = State.STANDBY
        self._swap_back = False


    def currentReward(self):
        return self.gs.currentReward()


def load_params(file_name,action_value_table):
    current_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_path,file_name)
    if os.path.getsize(file_path) <= 0:
        return

    file = open(file_path,'rb')
    controller._setParameters(pickle.load(file))
    print("Loading: " + str(controller.params))
def save_params(file_name,action_value_table):
    current_path = os.path.dirname(os.path.realpath(__file__))
    file = open(os.path.join(current_path,file_name),'wb')
    pickle.dump(controller.params,file)
    print("Saving: " + str(controller.params))
ROWS = 8
COLS = 8
WINDOW_SIZE = 4
GEM_TYPE_COUNT = 7
SPEED = 1
GAMESEED = 15
experiment = None
agent = None
OUTPUTFILE = "TrainedAIParams"
BATCH_SIZE = 2
LOAD = True
SAVE = True
if __name__ == "__main__":

    
    pygame.init()

    random.seed(GAMESEED) #Set static seed
    rand_state = random.getstate()
    fonts = load_fonts("Fonts")
    title_font = pygame.font.Font(fonts['OldSansBlack'],30)
    normal_font = pygame.font.Font(fonts['OldSansBlack'],15)
    sprites = load_sprites("Sprites")
    sounds = load_sounds("Sounds")
    
    num_states = 2**16
    num_actions = 24
    environment = Match3GUIEnvironment(ROWS,COLS,GEM_TYPE_COUNT,1,sprites,rand_state)
    controller = Match3ActionValueTable(num_states, num_actions)
    controller.initialize(1.)
    load_params(OUTPUTFILE,controller)
    learner = Q()
    agent = Match3Agent(controller, learner)
    task = Match3Task(environment)

    experiment = Match3Experiment(task, agent)
    environment.gs.print_board()
    environment.run()  
    try:
        if LOAD:
            load_params(OUTPUTFILE,controller)  
    except:
        pass
    
    if SAVE:
        save_params(OUTPUTFILE,controller)


    pygame.quit()
    quit()

