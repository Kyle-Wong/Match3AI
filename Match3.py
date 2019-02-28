from game import GameState
import pygame
import os.path

sprites = {}
def load_sprites(folder):
    '''
    Loads all sprites in the Match3AI/Sprites folder into the sprites dictionary.
    Key is file name (without extensions) and Value is a Surface of the sprite.
    '''
    current_path = os.path.dirname(os.path.realpath(__file__))
    sprite_path = os.path.join(current_path,"Sprites")
    for file in os.listdir(sprite_path):
        sprites[file.split('.')[0]] = pygame.image.load(os.path.join(sprite_path,file))
    print(sprites)
                            
class Match3:

    def __init__(self):
        self.gs = GameState(8,8,7)
        self.p_width = 800
        self.p_height = 600
        self.display = pygame.display.set_mode((self.p_width,self.p_height))
        self.surface = pygame.Surface((800,600))
        pygame.display.set_caption('Match3')
        self.clock = pygame.time.Clock()
        self.delta_time = 0

    def run(self):
        crashed = False
        while not crashed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    crashed = True

                #print(event)
            self.delta_time = self.clock.get_time()
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()
        quit()

    def update(self):
        pass
        
    def draw(self):
        self.surface = pygame.Surface((self.p_width,self.p_height))
        self.draw_background(60,0)
        self.draw_board(150,50,500,500)
        self.display.blit(self.surface,(0,0))
        
        

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
        

    def draw_background(self,offset_x = 0, offset_y = 0):
        x = self.surface.get_width()-sprites['background'].get_width()
        x /= 2
        y = self.surface.get_height()-sprites['background'].get_height()
        y/= 2
        x += offset_x
        y += offset_y
        self.surface.blit(sprites['background'],(x,y))

if __name__ == "__main__":
    load_sprites("Sprites")
    pygame.init()
    game = Match3()
    game.run()
    pygame.quit()
    quit()

