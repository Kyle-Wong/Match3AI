from game import GameState
from Gem import *
import pygame
import os.path

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


class Match3:

    def __init__(self,sprites):
        self.sprites = sprites
        self.gs = GameState(8,8,7)
        self.p_width = 800
        self.p_height = 600
        self.display = pygame.display.set_mode((self.p_width,self.p_height))
        self.board_rect = pygame.Rect(150,50,500,500)
        self.tile_w = self.board_rect.w/self.gs.cols
        self.tile_h = self.board_rect.h/self.gs.rows
        self.initialize_board()
        pygame.display.set_caption('Match3')
        self.surface = pygame.Surface((800,600))
        self.clock = pygame.time.Clock()
        self.delta_time = 0
        self.running = True

    def run(self):
        '''
        Begin main loop, call update and draw each frame
        '''
        while self.running:
            self.delta_time = self.clock.get_time()
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
                self.handle_key_down(event)

        
    def draw(self):
        '''
        Call the draw method of every object
        '''
        self.surface = pygame.Surface((self.p_width,self.p_height))
        self.draw_background(60,0)
        self.draw_board(self.board_rect.x,self.board_rect.y,self.board_rect.w,self.board_rect.h)
        self.draw_gems()
        self.display.blit(self.surface,(0,0))

    def handle_key_down(self,event):
        '''
        Handle all key inputs and do appropriate action
        '''
        if event.key == pygame.K_ESCAPE:
            self.running = False

    def draw_board(self,x,y,width,height):
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
    def draw_gems(self):
        for g in self.gems:
            g.draw(self.surface)
        

    def draw_background(self,offset_x = 0, offset_y = 0):
        '''
        Draw background image with optional x and y offset (defaults to centered image)
        '''
        x = (self.surface.get_width()-self.sprites['background'].get_width())/2
        y = (self.surface.get_height()-self.sprites['background'].get_height())/2
        x += offset_x
        y += offset_y
        self.surface.blit(self.sprites['background'],(x,y))

    def initialize_board(self):
        '''
        Spawn gem objects with correct sprite type
        '''
        self.gems = []
        for i in range(0,self.gs.rows):
            for j in range(0,self.gs.cols):
                self.gems.append(Gem(j,i,self.gs.board[i,j],3,self))

    
if __name__ == "__main__":
    sprites = load_sprites("Sprites")
    pygame.init()
    game = Match3(sprites)
    game.run()
    pygame.quit()
    quit()

