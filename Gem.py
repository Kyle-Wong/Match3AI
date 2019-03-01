import numpy as np
from Match3 import *
import math
def lerp(start,end,perc):
    return start + (end-start)*perc
class Gem:

    def __init__(self,row,col,gem_type,move_speed,game):
        self.row = row
        self.col = col
        self.gem_type = gem_type
        self.game = game
        self.x = self.game.board_rect.x+self.game.tile_w*self.col
        self.y = self.game.board_rect.y+self.game.tile_h*self.row
        self.move_speed = move_speed
        self.moving = False
        self.clear_anim = False
        self.rot_speed = 500
        self.angle = 0
        self.anim_duration = .25
        self.timer = 0

    def update(self):
        delta_x = self.game.board_rect.x+self.game.tile_w*self.col-self.x
        delta_y = self.game.board_rect.y+self.game.tile_h*self.row-self.y
        distance = math.hypot(delta_x,delta_y)
        angle = math.atan2(delta_y,delta_x)
        if self.clear_anim:
            self.timer += self.game.delta_time
            if self.timer > self.anim_duration:
                self.clear_anim = False
                self.angle= 0
                self.timer = 0
            else:
                self.angle += self.rot_speed*self.game.delta_time
        elif distance > self.move_speed*self.game.delta_time:
            self.x += math.cos(angle)*self.move_speed*self.game.delta_time
            self.y += math.sin(angle)*self.move_speed*self.game.delta_time
            self.moving = True
        else:
            self.x = self.game.board_rect.x+self.game.tile_w*self.col
            self.y = self.game.board_rect.y+self.game.tile_h*self.row
            self.moving = False

    def draw(self,screen):
        scalar = lerp(1,0,self.timer/self.anim_duration)
        sprite = self.game.sprites["gem"+str(self.gem_type)].convert_alpha()
        center = (self.x+self.game.tile_w/2,self.y+self.game.tile_h/2)
        rotated_img = pygame.transform.rotozoom(sprite, self.angle%360,scalar)
        screen.blit(rotated_img,pygame.Rect((int)(center[0]-rotated_img.get_width()/2),\
                                            (int)(center[1]-rotated_img.get_height()/2), \
                        sprite.get_width(),sprite.get_height()))

    def blocking(self):
        return self.moving or self.clear_anim
    
    def clear(self):
        self.clear_anim = True
        self.timer = 0
        self.angle = 0
        
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
        
    
