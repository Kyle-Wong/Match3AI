from game import GameState
from Gem import *
import random
import pygame
import os.path
from enum import Enum
class State(Enum):
    STANDBY = 1
    SWAP = 2
    FALL = 3

def load_sprites(folder):
    '''
    Loads all sprites in the Match3AI/Sprites folder into the sprites dictionary.
    Key is file name (without extensions) and Value is a Surface of the sprite.
    '''
    current_path = os.path.dirname(os.path.realpath(__file__))
    sprite_path = os.path.join(current_path,"Sprites")
    sprites = {}
    for file in os.listdir(sprite_path):
        sprites[file.split('.')[0]] = pygame.image.load(os.path.join(sprite_path,file))
    print("---------SPRITES----------")
    for s in sprites:
        print(s)
    return sprites

GAMESEED = 111
class Match3:

    def __init__(self,rows,cols,gem_type_count,turn_limit,sprites,rand_state = random.getstate()):
        self.sprites = sprites
        self.gs = GameState(rows,cols,gem_type_count,turn_limit,rand_state)
        self.p_width = 800
        self.p_height = 600
        self.display = pygame.display.set_mode((self.p_width,self.p_height))
        self.board_rect = pygame.Rect(150,0,500,500)
        self.tile_w = self.board_rect.w/self.gs.cols
        self.tile_h = self.board_rect.h/self.gs.rows
        self.objects = []
        self._initialize_board()
        pygame.display.set_caption('Match3')
        self.surface = pygame.Surface((800,600))
        self.clock = pygame.time.Clock()
        self.delta_time = 0
        self.running = True
        self.state = State.STANDBY
        self.prev_move = None

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
        if self.state is State.STANDBY:
            print("STANDBY")
        elif self.state is State.SWAP:
            if not self._gems_moving():
                remove_set = self.gs.get_matches()
                if len(remove_set) > 0:
                    self._match_gems(remove_set)
                    self.copy_board_to_gs()
                else:
                    self.state = State.STANDBY
            print("SWAPPING")
        elif self.state is State.FALL:
            if not self._gems_moving():
                remove_set = self.gs.get_matches()
                if len(remove_set) > 0:
                    self._match_gems(remove_set)
                    self.copy_board_to_gs()
                else:
                    self.state = State.STANDBY
        
    def draw(self):
        '''
        Call the draw method of every object
        '''
        self.surface = pygame.Surface((self.p_width,self.p_height))
        self._draw_background(60,0)
        self._draw_board(self.board_rect.x,self.board_rect.y,self.board_rect.w,self.board_rect.h)
        self._draw_gems()
        self._draw_prev_move()
        self.display.blit(self.surface,(0,0))

    def advance_state(self,p1,p2):
        '''
        Do a swap and move into SWAP
        Abort if state is not STANDBY
        '''
        if self.state is not State.STANDBY:
            return
        self.state = State.SWAP
        self.gs._swap(p1,p2)
        self._swap_gems(p1,p2)
        self.prev_move = (p1,p2)

    
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
        for i in range(self.gs.rows-1,-1,-1):
            for j in range(0,self.gs.cols):
                p1 = (i,j)
                if self.gems[p1].gem_type == -1:
                    p2 = self._get_above(p1)
                    if p2 is None:
                        self.gems[p1].jump_to_grid((h_stack[p1[1]],p1[1]))
                        self.gems[p1].gem_type = self.gs.generate_gem()
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
        
    def _gems_moving(self):
        '''
        Checks if any gems are currently in motion (false is all animation is finished)
        '''
        for i in range(0,self.gs.rows):
            for j in range(0,self.gs.cols):
                if not self.gems[i,j].on_tile:
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

    def _initialize_board(self):
        '''
        Spawn gem objects with correct sprite type
        '''
        self.gems = np.empty((self.gs.rows,self.gs.cols),dtype=Gem)
        for i in range(0,self.gs.rows):
            for j in range(0,self.gs.cols):
                self.gems[i,j] = Gem(i,j,self.gs.board[i,j],200,self)
                self.objects.append(self.gems[i,j])
    def copy_board_to_gs(self):
        for i in range(0,self.gs.rows):
            for j in range(0,self.gs.cols):
                self.gs.board[i,j] = self.gems[i,j].gem_type

    
if __name__ == "__main__":
    sprites = load_sprites("Sprites")
    pygame.init()
    random.seed(GAMESEED)
    state = random.getstate()
    game = Match3(8,8,7,10,sprites,state)
    game.run()
    pygame.quit()
    quit()

