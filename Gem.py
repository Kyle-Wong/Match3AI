import numpy as np
from Match3 import *
import math
class Gem:

    def __init__(self,row,col,gem_type,move_speed,game):
        self.row = row
        self.col = col
        self.gem_type = gem_type
        self.game = game
        self.x = self.game.board_rect.x+self.game.tile_w*self.col
        self.y = self.game.board_rect.y+self.game.tile_h*self.row
        self.move_speed = move_speed
        self.on_tile = True

    def update(self):
        delta_x = self.game.board_rect.x+self.game.tile_w*self.col-self.x
        delta_y = self.game.board_rect.y+self.game.tile_h*self.row-self.y
        distance = math.hypot(delta_x,delta_y)
        angle = math.atan2(delta_y,delta_x)
        if distance > self.move_speed*self.game.delta_time:
            self.x += math.cos(angle)*self.move_speed*self.game.delta_time
            self.y += math.sin(angle)*self.move_speed*self.game.delta_time
            self.on_tile = False
        else:
            self.x = self.game.board_rect.x+self.game.tile_w*self.col
            self.y = self.game.board_rect.y+self.game.tile_h*self.row
            self.on_tile = True

    def draw(self,screen):
        sprite = self.game.sprites["gem"+str(self.gem_type)]
        x_offset = (self.game.tile_w-sprite.get_width())/2
        y_offset = (self.game.tile_h-sprite.get_height())/2
        screen.blit(sprite,pygame.Rect((int)(self.x+x_offset),(int)(self.y+y_offset), \
                        sprite.get_width(),sprite.get_height()))

    def set_grid(self,grid_tile):
        if (self.row,self.col) == grid_tile:
            #No movement, do not set on_tile to False
            return
        self.row = grid_tile[0]
        self.col = grid_tile[1]
        self.on_tile = False
        
    def set_pos(self,point):
        self.x = point[0]
        self.y = point[1]
        self.on_tile = False

    def jump_to_grid(self,grid_tile):
        self.x = self.game.board_rect.x+self.game.tile_w*grid_tile[1]
        self.y = self.game.board_rect.y+self.game.tile_h*grid_tile[0]
        
    
