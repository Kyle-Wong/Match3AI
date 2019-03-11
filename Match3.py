from game import GameState
from Gem import *
from TFRLearningMatch3 import *
import random
import pygame
import os.path
from enum import Enum
import tensorflow as tf
import numpy as np
import math
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

GAMESEED = 111 #Static random number generator seed
TRAINING_ROUNDS = 100 #Display GUI after X training rounds
GAME_RUNNING = False
class Match3:

    def __init__(self,rows,cols,gem_type_count,turn_limit,sprites,gr,rand_state = random.getstate()):
        GAME_RUNNING = True
        self.gr = gr
        self.sprites = sprites
        self.p_width = 930 #Pixel width
        self.p_height = 600 #Pixel height

        self.gs = GameState(rows,cols,gem_type_count,turn_limit,self.gr._env.rand_state) #GameState
        gr.run(False)
        
        self.action_queue = gr._action_queue
                                #list of pairs ( (p1.x,p1.y), (p2.x,p2.y) ) to be
                                #run sequentially as swaps in the game
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
        
        
    def run(self):
        '''
        Begin main loop, call update and draw each frame
        '''
        
        
        while self.running:
            self.delta_time = self.clock.get_time()/1000
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
        self._update_objects()
        if self.gs.turn_num >= self.gs.turn_limit or not self.gs._calculate_if_moves_left():
            self.__init__(8,8,7,10,sprites,self.gr,self.gs.rand_state)
        if self.state is State.STANDBY:
            if not self._gems_blocking() and len(self.action_queue) > 0:
                p1,p2 = self.action_queue[0]
                self.advance_state(p1,p2)
                self.action_queue.pop(0)
                
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
        self.display.blit(self.surface,(0,0))

    def advance_state(self,p1,p2):
        '''
        Do a swap and move into SWAP
        Abort if state is not STANDBY
        '''
        if self.state is not State.STANDBY:
            return
        self.state = State.CLEARING
        self.prev_matches = self.gs.gems_matched
        self.gs._swap(p1,p2)
        self._swap_gems(p1,p2)
        self.prev_move = (p1,p2)
        if len(self.gs.get_matches()) == 0:
            self._swap_back = True

        self.gs.turn_num += 1

    def settle_step(self):
        remove_set = self.gs.get_matches()
        if len(remove_set) > 0:
            self.copy_board_to_gs()
            self._match_gems(remove_set)
            self.copy_board_to_gs()
            self.gs.gems_matched += len(remove_set)
            self.gs.score += len(remove_set)
            self.state = State.CLEARING
        else:
            self.state = State.STANDBY
            self.improvement = self.gs.gems_matched-self.prev_matches
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
            self.advance_state((6,3),(6,4))
        if event.key == pygame.K_r:
            self.__init__(8,8,7,10,sprites,state)
    def _update_objects(self):
        for o in self.objects:
            o.update()
    def _draw_board(self,x,y,width,height):
        '''
        Draw game board background
        '''
        step_w = (int)(width/self.gs.cols)
        step_h = (int)(height/self.gs.rows)
        self.sub_surface = pygame.Surface((step_w,step_h))
        color = pygame.Color(0,0,0)
        for i in range(0,self.gs.rows):
            for j in range(0,self.gs.cols):
                rect = pygame.Rect(x+j*step_w,y+i*step_h,step_w,step_h)
                self.sub_surface.fill(pygame.Color(30,30,30))
                if (i+j)%2 == 0:
                    self.sub_surface.set_alpha(150)
                else:
                    self.sub_surface.set_alpha(90)
                self.surface.blit(self.sub_surface,rect)
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
        text = "Moves Left:"
        surface = title_font.render(text,True,pygame.Color(0,0,0))
        text_dim = title_font.size(text)
        self._draw_text_outline(text,x,y,text_dim[0],text_dim[1],1.25,font,pygame.Color(255,255,255))
        self.surface.blit(surface,pygame.Rect(x,y,text_dim[0],text_dim[1]))

        text = str(self.gs.turn_limit-self.gs.turn_num)
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

        text = str(self.gs.gems_matched*100)
        surface = font.render(text,True,pygame.Color(0,0,0))
        text_dim = font.size(text)
        self._draw_text_outline(text,x,y+40,text_dim[0],text_dim[1],1.25,font,pygame.Color(255,255,255))
        self.surface.blit(surface,pygame.Rect(x,y+40,text_dim[0],text_dim[1]))
    def _draw_credits(self,x,y,font):
        text = "Gems by limbusdev, Sounds by ViRiX Dreamcore (soundcloud.com/virix)"
        surface = font.render(text,True,pygame.Color(0,0,0))
        text_dim = font.size(text)
        self.surface.blit(surface,pygame.Rect(x,y,text_dim[0],text_dim[1]))

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



if __name__ == "__main__":

    
    random.seed(GAMESEED) #Set static seed
    state = random.getstate()
    env = GameState(8, 8, 7, 10,state)

    num_states = env.cols * env.rows
    num_actions = env.cols * (env.rows - 1) + env.rows * (env.cols - 1)

    model = Model(num_states, num_actions, BATCH_SIZE)
    mem = Memory(50000)
    with tf.Session() as sess:
        sess.run(model.var_init)
        gr = GameRunner(sess, model, env, mem, MAX_EPSILON, MIN_EPSILON, LAMBDA)
        num_episodes = 100 #10000
        plot_interval = num_episodes
        cnt = 0
        while cnt < num_episodes:
            gr.run(cnt >= num_episodes - 1)
            cnt += 1


        pygame.init()
        fonts = load_fonts("Fonts")
        title_font = pygame.font.Font(fonts['OldSansBlack'],30)
        normal_font = pygame.font.Font(fonts['OldSansBlack'],15)
        sprites = load_sprites("Sprites")
        sounds = load_sounds("Sounds")
        state = gr._env.rand_state
        game = Match3(8,8,7,10,sprites,gr,state)
        game.run()
        
            
    pygame.quit()
    quit()

