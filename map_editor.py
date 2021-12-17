import pygame as pg
import numpy as np
import json
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
class O_object(object):
    def __init__(self, name, location = [0, 0]):
        self.name = name
        self.location = location

class Creator():
    def __init__(self, map_size=[10, 10], tile_size = 120):
        self.tile_size = tile_size # 40
        self.map_size = map_size
        self.button_size = 80
        self.button_num = 3
        self.selected_button = tuple([map_size[0],0])
        self.exit = False
        self.mouse_pos = []
        self.width_height = [x * self.tile_size for x in self.map_size]
        self.drawn_objects = []
        self.button_objects = []
        self.button_names = ['T_is', 'T_baba', 'T_flag',  
                            'T_rock', 'T_wall', 'T_skull', 'T_water', 'T_lava',
                            'O_baba', 'O_flag', 'O_rock', 'O_wall',  'O_skull', 'O_water', 'O_lava',
                            'TR_you', 'TR_win', 'TR_stop', 'TR_push', 'TR_defeat', 'TR_sink', 'TR_hot', 'TR_melt'
                            ]
        
        self.drag = False
        self.sprites = {
            'background':(8, 8, 8),
            'line':(100, 100, 100),
            'selected':pg.image.load('baba_imgs/selected.png'),
            'save':pg.image.load('baba_imgs/save.png'),
            'reset':pg.image.load('baba_imgs/reset.png'),
            'O_MW':pg.image.load('baba_imgs/O_MW.png'),
            'T_is':pg.image.load('baba_imgs/T_is.png'),
            'O_baba':pg.image.load('baba_imgs/O_baba.png'),
            'T_baba':pg.image.load('baba_imgs/T_baba.png'),
            'O_rock':pg.image.load('baba_imgs/O_rock.png'),
            'T_rock':pg.image.load('baba_imgs/T_rock.png'),
            'O_flag':pg.image.load('baba_imgs/O_flag.png'),
            'T_flag':pg.image.load('baba_imgs/T_flag.png'),
            'O_wall':pg.image.load('baba_imgs/O_wall.png'),
            'T_wall':pg.image.load('baba_imgs/T_wall.png'),
            'O_skull':pg.image.load('baba_imgs/O_skull.png'),
            'T_skull':pg.image.load('baba_imgs/T_skull.png'),
            'O_lava':pg.image.load('baba_imgs/O_lava.png'),
            'T_lava':pg.image.load('baba_imgs/T_lava.png'),
            'O_water':pg.image.load('baba_imgs/O_water.png'),
            'T_water':pg.image.load('baba_imgs/T_water.png'),
            'TR_you':pg.image.load('baba_imgs/TR_you.png'),
            'TR_win':pg.image.load('baba_imgs/TR_win.png'),
            'TR_stop':pg.image.load('baba_imgs/TR_stop.png'),
            'TR_push':pg.image.load('baba_imgs/TR_push.png'),
            'TR_defeat':pg.image.load('baba_imgs/TR_defeat.png'),
            'TR_sink':pg.image.load('baba_imgs/TR_sink.png'),
            'TR_hot':pg.image.load('baba_imgs/TR_hot.png'),
            'TR_melt':pg.image.load('baba_imgs/TR_melt.png')
        }

        pg.init()

        self.width_height[0] += self.button_size * self.button_num
        self.screen = pg.display.set_mode(tuple(self.width_height))
        self.width_height[0] -= self.button_size * self.button_num
        self.reset()

    def reset(self):
        self.button_objects = []
        self.drawn_objects = []
        x = 0
        y = 0
        for name in self.button_names: 
            
            if x == self.button_num:
                y += 1
                x = 0
            pos = tuple([self.map_size[0]+x, y])
            x += 1
            self.button_objects.append(O_object(name, pos))
    

    def draw(self):
        self.screen.fill(self.sprites['background'])
        button = pg.transform.scale(self.sprites['selected'],(self.button_size, self.button_size))
        self.screen.blit(button,self.grid_to_raw(self.selected_button))
        button = pg.transform.scale(self.sprites['reset'],(self.button_size, self.button_size))
        self.screen.blit(button,self.grid_to_raw(tuple([self.map_size[0]+self.button_num-1,(self.width_height[1]//self.button_size)-1])))
        button = pg.transform.scale(self.sprites['save'],(self.button_size, self.button_size))
        self.screen.blit(button,self.grid_to_raw(tuple([self.map_size[0]+self.button_num-1,(self.width_height[1]//self.button_size)-2])))

        for x in range(0, self.width_height[0]+self.tile_size, self.tile_size):
            pg.draw.line(self.screen, self.sprites['line'], (x, 0), (x, self.width_height[1]))
        for y in range(0, self.width_height[1], self.tile_size):
            pg.draw.line(self.screen, self.sprites['line'], (0, y), (self.width_height[0], y))
        
        for item in self.button_objects:
            sprite = pg.transform.scale(self.sprites[item.name],(self.button_size, self.button_size))
            location = self.grid_to_raw(item.location)
            self.screen.blit(sprite,location)

        for item in self.drawn_objects:
            sprite = pg.transform.scale(self.sprites[item.name],(self.tile_size, self.tile_size))
            location = self.grid_to_raw(item.location)
            self.screen.blit(sprite,location)

        pg.display.flip()

    def raw_to_grid(self, pos):
        if self.width_height[0] <= pos[0]:
            tpw = self.map_size[0] + (pos[0]%self.width_height[0])//self.button_size
            tph = pos[1] // self.button_size
            pos = (tpw,tph)
            self.selected_button = pos
        else:
            pos = tuple([x//self.tile_size for x in pos])
        return pos

    def grid_to_raw(self, pos):
        if self.map_size[0] <= pos[0]:
            tpw = self.map_size[0] * self.tile_size + \
                (pos[0] - self.map_size[0]) * self.button_size
            tph = pos[1] * self.button_size
            pos = tuple([tpw, tph])
        else:
            pos = tuple([x * self.tile_size for x in pos])

        return pos

    def save(self):
        names = []
        locations = []
        for object_ in self.drawn_objects:
            names.append(object_.name)
            locations.append([sum(x) for x in zip(object_.location, [1,1])])

        data = {}
        data['names'] = names
        data['locations'] = locations
        data['map_size'] = self.map_size
        
        filename = asksaveasfilename(initialdir = " ",title = "choose your file",filetypes = (("JSON files","*.json"),("all files","*.*")))
        with open(filename+'.json', 'w') as outfile:
            json.dump(data, outfile, indent=4)

    def change_selected_button(self,move):
        tmp = [sum(x) for x in zip(self.selected_button, move)]
        if tmp[0]>= self.map_size[0]+self.button_num-1:
            tmp[0] = self.map_size[0]+self.button_num-1
        if tmp[0] <= self.map_size[0]:
            tmp[0] = self.map_size[0]
        if tmp[1] >= (self.width_height[1]//self.button_size)-3:
            tmp[1] = (self.width_height[1]//self.button_size)-3
        if tmp[1] <= 0:
            tmp[1] = 0
        self.selected_button = tuple(tmp)


    def main_loop(self):
        while not self.exit:
            self.draw()
            events = pg.event.get()
            for event in events:
                if event.type == pg.KEYDOWN:
                    if event.unicode == 'w': # w
                        self.change_selected_button([0,-1])
                    if event.unicode == 's':# s
                        self.change_selected_button([0,1])
                    if event.unicode == 'a':# a
                        self.change_selected_button([-1,0])
                    if event.unicode == 'd':# d
                        self.change_selected_button([1,0])

                if event.type == pg.MOUSEBUTTONDOWN:
                    self.drag = True
                if event.type == pg.MOUSEBUTTONUP:
                    self.drag = False
                    self.mouse_pos=[]

                if self.drag:
                    draw = True
                    pos = pg.mouse.get_pos()
                    pos = self.raw_to_grid(pos)
                    if pos != self.mouse_pos:
                        for object_ in self.drawn_objects:
                            if object_.location == pos:
                                draw = False
                                self.mouse_pos = pos
                                self.drawn_objects.remove(object_)
                        if draw and pos != self.mouse_pos:
                            self.mouse_pos = pos
                            if pos[0] < self.map_size[0]:
                                for object_ in self.button_objects:
                                    if self.selected_button == object_.location:
                                        self.drawn_objects.append(O_object(object_.name,pos))
                    
                    # print(pos, self.selected_button)
                    if self.selected_button == tuple([self.map_size[0]+self.button_num-1,(self.width_height[1]//self.button_size)-1]):
                        self.reset() 
                        self.selected_button = tuple([self.map_size[0],0])

                    if self.selected_button == tuple([self.map_size[0]+self.button_num-1,(self.width_height[1]//self.button_size)-2]):
                        self.selected_button = tuple([self.map_size[0],0])
                        self.save() 
                    

creator = Creator()
creator.main_loop()