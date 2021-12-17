from glob import glob
import json
import copy
# TODO list to deque or tuple
from collections import deque

# import gym TODO use gym wraaper
import numpy as np
import cv2
import pygame as pg

from util import time_measure

def load_stages_from_files(only_this_state=10000, path_to_stages='stages/'):
    names = deque()
    locations = deque()
    map_sizes = deque()
    cnt = 0
    for file_name in glob(path_to_stages+'*.json'):
        with open(file_name) as f:
            temp = json.load(f)
            names.append(temp['names'])
            locations.append(temp['locations'])
            map_sizes.append(temp['map_size'])
            
            temp.clear()
            f.close()
        if only_this_state == cnt:
            break

        if only_this_state != 10000:
            names = deque()
            locations = deque()
            map_sizes = deque()
        cnt += 1
    return names, locations, map_sizes

def get_keyboard_input():
    action = 0
    events = pg.event.get()
    for event in events:
        if event.type == pg.KEYDOWN:
            if event.key == 119 or event.key == 273 : # up
                action = 1
            if event.key == 115 or event.key == 274:# down
                action = 2
            if event.key == 97 or event.key == 276:# left
                action = 3
            if event.key == 100 or event.key == 275:# right
                action = 4
            if event.key == 114:# d
                action = 'r'
    return action

class Engine():
    def __init__(self):
        self.rules = {}
        self.objects = deque()
        self.boundary = deque()
        self.action_list = deque(((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0),
                                (0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)))

        
    def add(self, name_list):
        for name in name_list:
            stop = 0
            is_text = 0
            if name == 'O_MW':
                stop = 1
            if name.find('T_') == 0 or name.find('TR_') == 0:
                is_text = 1
            self.rules[name] = {'you' : 0,
                                'win' : 0,
                                'push' : is_text,
                                'stop' : stop,
                                'defeat' : 0,
                                'sink' : 0,
                                'hot' : 0,
                                'melt' : 0,
                                'lock' : 0
                                }
    def clear(self):
        del self.rules
        self.rules = {}
        
    def get_names_got(self, arg):
        name_list  = deque()
        for key in self.rules:
            cnt = 0
            for (_, _) in set(self.rules[key].items()) & set(arg.items()):
                cnt = cnt + 1
            if cnt == len(arg):
                name_list.append(key)
        # if arg == 'stop':
        # else:
        #     for key in self.rules:
        #         if self.rules[key][attribute] == 1 and self.rules[key]['stop'] == 0:
        #             name_list.append(key)
        return name_list

    def effecting_on_target(self, all_is_all):
        for target, effect in all_is_all:
            try:
                if effect.name.find('T_') == 0:
                    if target.point == effect.point:
                        self.rules[target.point]['lock'] = 1
            except:
                pass

        for target, effect in all_is_all:
            try:
                if effect.name.find('TR_') == 0:
                    self.rules[target.point][effect.point] = 1
                elif effect.name.find('T_') == 0:
                    if self.rules[target.point]['lock'] == 0:
                        for item in self.objects:
                            if target.point == item.name:
                                item.name = effect.point
            except:
                pass
    
    def move_object_to(self, action, col_locations, push, state):
        if len(col_locations) == 0:
            return col_locations
        for col_loc in col_locations:
            for push_object in list(push):
                if push_object.location == col_loc:
                    tmp_lo = [sum(x) for x in zip(push_object.location, self.action_list[action])]
                    real_lo = self.move_object_to(action, [tmp_lo], push, state)
                    push_object.location = real_lo
                    push.remove(push_object)
        return col_locations[0]

    def delete_crashed_objects(self, col_locations, defeat, state, delete_single_object=False):
        if not delete_single_object:
            for col_loc in col_locations:
                for defeat_object in list(defeat):
                    if defeat_object.location == col_loc:
                        state.remove(defeat_object)
                        defeat.remove(defeat_object)
        else:
            for col_object in col_locations:
                for defeat_object in list(defeat):
                    if defeat_object.location == col_object.location:
                        state.remove(defeat_object)
                        state.remove(col_object)
                        defeat.remove(defeat_object)
                        break
        return 0

    def what_is_this(self, arg, object_list):
        objects = deque()
        names = self.get_names_got(arg)
        for object_ in object_list:
            try:
                names.index(object_.name)
                objects.append(object_)
            except:
                pass
        return objects

    def where_is_this(self, arg, object_list):
        locations = deque()
        names = self.get_names_got(arg)
        for object_ in object_list:
            try:
                names.index(object_.name)
                locations.append(object_.location)
            except:
                pass
        return locations
    

class O_object(object):
    def __init__(self, name, location = [0, 0]):
        self.name = name
        self.location = location
        self.point = None
        self.is_text = 0
        if name.find('TR_')==0:
            self.point  = name[3:len(name)]
            self.is_text = 1
        if name.find('T_')==0:
            self.point = 'O_'+name[2:len(name)]
            self.is_text = 1

class Env():
    def __init__(self, starting_stage=0, training_on_single_stage = False, tile_size = 40):#name_list, location_list, map_size, starting_stage=0, tile_size = 40):
        if training_on_single_stage == False:
            self.all_namelist, locations, self.all_mapsize = load_stages_from_files()
            self.all_stages = list(zip(self.all_namelist, locations))
            self.cuurent_stage = starting_stage
        else:
            self.all_namelist, locations, self.all_mapsize = load_stages_from_files(starting_stage)
            self.all_stages = list(zip(self.all_namelist, locations))
            self.cuurent_stage = 0

        self.engine = Engine()
        self.name_list, self.location_list, self.map_size = None, None, None

        self.select_stage()
        
        self.tile_size = tile_size # 40

        self.grid_width_height = [x + 2 for x in self.map_size]
        self.width_height = tuple([x * self.tile_size for x in self.grid_width_height])
        self.sprites = {
            'O_MW':deque([cv2.imread('baba_imgs/O_MW.png',cv2.IMREAD_UNCHANGED),0]),
            'T_is':deque([cv2.imread('baba_imgs/T_is.png',cv2.IMREAD_UNCHANGED),100]),
            'O_baba':deque([cv2.imread('baba_imgs/O_baba.png',cv2.IMREAD_UNCHANGED),1]),
            'T_baba':deque([cv2.imread('baba_imgs/T_baba.png',cv2.IMREAD_UNCHANGED),101]),
            'O_rock':deque([cv2.imread('baba_imgs/O_rock.png',cv2.IMREAD_UNCHANGED),2]),
            'T_rock':deque([cv2.imread('baba_imgs/T_rock.png',cv2.IMREAD_UNCHANGED),102]),
            'O_flag':deque([cv2.imread('baba_imgs/O_flag.png',cv2.IMREAD_UNCHANGED),3]),
            'T_flag':deque([cv2.imread('baba_imgs/T_flag.png',cv2.IMREAD_UNCHANGED),103]),
            'O_wall':deque([cv2.imread('baba_imgs/O_wall.png',cv2.IMREAD_UNCHANGED),4]),
            'T_wall':deque([cv2.imread('baba_imgs/T_wall.png',cv2.IMREAD_UNCHANGED),104]),
            'O_skull':deque([cv2.imread('baba_imgs/O_skull.png',cv2.IMREAD_UNCHANGED),5]),
            'T_skull':deque([cv2.imread('baba_imgs/T_skull.png',cv2.IMREAD_UNCHANGED),105]),
            'O_lava':deque([cv2.imread('baba_imgs/O_lava.png',cv2.IMREAD_UNCHANGED),6]),
            'T_lava':deque([cv2.imread('baba_imgs/T_lava.png',cv2.IMREAD_UNCHANGED),106]),
            'O_water':deque([cv2.imread('baba_imgs/O_water.png',cv2.IMREAD_UNCHANGED),7]),
            'T_water':deque([cv2.imread('baba_imgs/T_water.png',cv2.IMREAD_UNCHANGED),107]),
            'TR_you':deque([cv2.imread('baba_imgs/TR_you.png',cv2.IMREAD_UNCHANGED),200]),
            'TR_win':deque([cv2.imread('baba_imgs/TR_win.png',cv2.IMREAD_UNCHANGED),201]),
            'TR_stop':deque([cv2.imread('baba_imgs/TR_stop.png',cv2.IMREAD_UNCHANGED),202]),
            'TR_push':deque([cv2.imread('baba_imgs/TR_push.png',cv2.IMREAD_UNCHANGED),203]),
            'TR_defeat':deque([cv2.imread('baba_imgs/TR_defeat.png',cv2.IMREAD_UNCHANGED),204]),
            'TR_sink':deque([cv2.imread('baba_imgs/TR_sink.png',cv2.IMREAD_UNCHANGED),205]),
            'TR_hot':deque([cv2.imread('baba_imgs/TR_hot.png',cv2.IMREAD_UNCHANGED),206]),
            'TR_melt':deque([cv2.imread('baba_imgs/TR_melt.png',cv2.IMREAD_UNCHANGED),207])
        }
        patcher = np.zeros((self.tile_size,self.tile_size,3),np.uint8)
        cv2.line(patcher, (0, 0),(0, self.tile_size), (31, 21, 24), 2, 2)
        cv2.line(patcher, (self.tile_size, 0),(self.tile_size, self.tile_size), (31, 21, 24), 2, 2)
        cv2.line(patcher, (0, 0),(self.tile_size, 0), (31, 21, 24), 2, 2)
        cv2.line(patcher, (0, self.tile_size),(self.tile_size, self.tile_size), (31, 21, 24), 2, 2)

        for key in self.sprites:
            self.sprites[key][0] = cv2.resize(self.sprites[key][0],(self.tile_size, self.tile_size), interpolation=cv2.INTER_NEAREST)
            alpha = self.sprites[key][0][:, :, 3]/255.0
            reversed_alpha = 1.0-alpha
            for c in range(0,3):
                self.sprites[key][0][:,:,c] = alpha * self.sprites[key][0][:,:,c] + reversed_alpha * patcher[:,:,c]

        self.arrays = None
        self.images = None
        self.observation = None
        
        pg.init()

    def select_stage(self):
        self.name_list = self.all_namelist[self.cuurent_stage]+['O_MW']
        self.location_list = list(zip(*self.all_stages[self.cuurent_stage]))
        self.map_size = self.all_mapsize[self.cuurent_stage]

    def generate_observation(self): # cv2 = BGR
        self.images = np.zeros((self.width_height[0],self.width_height[1],3),np.uint8)
        
        for x in range(0, self.width_height[0], self.tile_size):
            cv2.line(self.images, (int(x), 0),(int(x), self.width_height[1]), (31, 21, 24), 2, 2)
        for y in range(0, self.width_height[1], self.tile_size):
            cv2.line(self.images, (0, int(y)),(self.width_height[0], int(y)), (31, 21, 24), 2, 2)
        
        objects = self.engine.objects + self.engine.boundary

        for item in objects:
            location = [x*self.tile_size for x in item.location]
            # print(item.name, item.location)
            self.images[location[1]:location[1]+self.tile_size, location[0]:location[0]+self.tile_size] = self.sprites[item.name][0][:,:,:3]

        self.arrays = np.zeros((9,self.map_size[1],self.map_size[0]))
        
        objects = self.engine.objects

        for item in objects:
            obs_loc = [item.location[1]-1,item.location[0]-1] # item.loc[0] --> x val | item.loc[1] --> y val
            self.arrays[0,obs_loc[0],obs_loc[1]] = self.sprites[item.name][1] # object_number and locations
            self.arrays[1:,obs_loc[0],obs_loc[1]] = np.array(list(self.engine.rules[item.name].items()))[:-1,1].astype(np.float) # object_properties
        
        # self.images = np.dot(self.images[...,:3], [0.2989, 0.5870, 0.1140])
        self.observation = [self.images, self.arrays]

    def render(self):
        # rgb = bgr[...,::-1]
        # bgr = rgb[...,::-1]
        # gbr = rgb[...,[2,0,1]]
        self.screen = pg.display.set_mode(self.width_height)
        self.screen.blit(pg.surfarray.make_surface(np.rot90(np.flip(self.images[...,::-1],1))), (0,0))
        pg.display.flip()
        # for event in pg.event.get():
        #     if event.type == pg.QUIT:
        #         pg.display.quit()
        #         pg.quit()
        # if rendering == False:
        #     cv2.destroyAllWindows()
        # else:
        #     cv2.imshow('baba_is_gym',self.images)
        #     cv2.waitKey(1)

    def get_random_action(self, num = 5):
        return np.random.randint(num)

    def step_from_here(self,action,state):
        self.engine.objects = copy.deepcopy(state)
        s, r, d, i = self.step(action)
        return s, r, d, i
    

    def step(self, action):
        state = self.engine.objects
        reward = 0
        done = 0
        t = time_measure()
        if action != 0:
            # move 'you' objects first
            yous = self.engine.what_is_this({'you' : 1}, state)
            for you in yous:
                you.location = [sum(x) for x in zip(you.location, self.engine.action_list[action])]

            # move 'push' objects
            you_loc = self.engine.where_is_this({'you' : 1}, state)
            push = self.engine.what_is_this({'you' : 0, 'push' : 1}, state)
            self.engine.move_object_to(action, you_loc, push, state)

            # undo if collision with 'stop' object
            stop = self.engine.where_is_this({'you' : 0, 'push' : 0, 'stop' : 1}, self.engine.boundary+state)
            push = self.engine.what_is_this({'push' : 1}, state)
            self.engine.move_object_to(action+5, stop, push, state)

            # undo single 'you' object that collision with other 'push' or 'stop' object
            yous = self.engine.what_is_this({'you' : 1}, state)
            push = self.engine.where_is_this({'you' : 0, 'push' : 1}, state)
            stop = self.engine.where_is_this({'you' : 0, 'stop' : 1}, self.engine.boundary+state)
            for you in yous:
                if you.location in push:
                    you.location = [sum(x) for x in zip(you.location, self.engine.action_list[action+5])]
                    continue
                if you.location in stop:
                    you.location = [sum(x) for x in zip(you.location, self.engine.action_list[action+5])]
                    continue

            # update state
            self.update_attributes()

            # delete object if crash with defeat object
            defeat = self.engine.where_is_this({'defeat' : 1}, state)
            you = self.engine.what_is_this({'you' : 1}, state)
            self.engine.delete_crashed_objects(defeat, you, state)
            hot = self.engine.where_is_this({'hot' : 1}, state)
            melt = self.engine.what_is_this({'melt' : 1}, state)
            self.engine.delete_crashed_objects(hot, melt, state)
            sink = self.engine.what_is_this({'sink' : 1}, state)
            all_except_sink = self.engine.what_is_this({'sink' : 0}, state)
            self.engine.delete_crashed_objects(sink, all_except_sink, state, delete_single_object=True)
        
        # update state
        self.update_attributes()

        # selecting reward
        win = self.engine.where_is_this({'win' : 1}, state)
        you = self.engine.where_is_this({'you' :1}, state)
        
        if len(you) == 0:
            reward = -10
            done = 1      

        for win_lo in win:
            try:
                you.index(win_lo)
                reward = 10
                done = 1
                self.cuurent_stage += 1
                self.cuurent_stage = self.cuurent_stage % len(self.all_namelist)
            except:
                pass

        self.generate_observation()
        state = copy.deepcopy(self.observation)
        info = copy.deepcopy(self.engine.objects)
        return state, reward, done, info

    def clear_attributes(self):
        self.engine.clear()
        self.engine.add(self.name_list)

    def update_attributes(self):
        self.clear_attributes()

        is_list = deque()
        for item in self.engine.objects:
            if item.name == 'T_is':
                is_list.append(item.location)
        # [[x[0]-1, x[1]] for x in is_list]
        h_is_l = [[sum(pairs) for pairs in zip(*pair, [-1, 0])] for pair in zip(is_list)]      
        h_is_r = [[sum(pairs) for pairs in zip(*pair, [1, 0])] for pair in zip(is_list)]
        v_is_u = [[sum(pairs) for pairs in zip(*pair, [0, -1])] for pair in zip(is_list)]
        v_is_d = [[sum(pairs) for pairs in zip(*pair, [0, 1])] for pair in zip(is_list)]
        
        for obj in self.engine.objects:
            if obj.is_text == 1 and obj.name != 'T_is':
                try:
                    h_is_l[h_is_l.index(obj.location)] = obj
                except:
                    pass
                try:    
                    h_is_r[h_is_r.index(obj.location)] = obj
                except:
                    pass
                try:
                    v_is_u[v_is_u.index(obj.location)] = obj
                except:
                    pass
                try:
                    v_is_d[v_is_d.index(obj.location)] = obj
                except:
                    pass
        
        all_is_all = list(zip(h_is_l,h_is_r)) + list(zip(v_is_u,v_is_d))
        self.engine.effecting_on_target(all_is_all)

    def reset(self):
        self.select_stage()
        obj_list = deque()
        border_list = deque()
        names_locations = self.location_list
        map_size = self.map_size
        
        self.engine.clear()
        self.engine.add(self.name_list)

        for i in range(map_size[0]+2):
            border_list.append(O_object('O_MW',[i,0]))
            border_list.append(O_object('O_MW',[i,map_size[1]+1]))

        for i in range(1, map_size[1]+1):
            border_list.append(O_object('O_MW',[0,i]))
            border_list.append(O_object('O_MW',[map_size[0]+1,i]))
        
        
        for name, location in names_locations:
            obj_list.append(O_object(name, location))
        
        self.engine.objects = obj_list
        self.engine.boundary = border_list
        
        state, reward, done, info = self.step(0)
        return state , info