import numpy as np
from Match3 import *
import math
class Gem:

    def __init__(self,grid_x,grid_y,gem_type,move_speed,game):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.gem_type = gem_type
        self.game = game
        self.x = self.game.board_rect.x+self.game.tile_w*self.grid_x
        self.y = self.game.board_rect.y+self.game.tile_h*self.grid_y
        self.move_speed = move_speed

    def update(self):
        delta_x = self.game.board_rect.x+self.game.tile_w*self.grid_x-self.x
        delta_y = self.game.board_rect.y+self.game.tile_h*self.grid_y-self.y
        distance = math.hypot(delta_x,delta_y)
        angle = math.atan2(delta_y,delta_x)
        if distance > self.move_speed*self.gs.delta_time:
            self.x += math.cos(angle)*self.move_speed*self.game.delta_time
            self.y += math.sin(angle)*self.move_speed*self.game.delta_time
        else:
            self.x = self.game.board_rect.x+self.game.tile_w*self.grid_x
            self.y = self.game.board_rect.y+self.game.tile_h*self.grid_y

    def draw(self,screen):
        sprite = self.game.sprites["gem"+str(self.gem_type)]
        x_offset = (self.game.tile_w-sprite.get_width())/2
        y_offset = (self.game.tile_h-sprite.get_height())/2
        screen.blit(sprite,pygame.Rect((int)(self.x+x_offset),(int)(self.y+y_offset), \
                        sprite.get_width(),sprite.get_height()))
        
    
