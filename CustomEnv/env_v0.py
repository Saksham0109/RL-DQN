import numpy as np
import random
import cv2
from gym import Env, spaces
import torch

class Box(object):
    '''To store attributes of a box'''

    def __init__(self, pos, id):
        self.id = id
        self.pos = pos


class Agent(object):
    '''To store attributes of a drone'''

    def __init__(self, pos, id):
        self.id = id
        self.pos = pos
        self.catch_box = -1


class EnvPlot(Env):
    def __init__(self, agt_num, box_num, dim=(25, 25)):
        '''Initialize the environment'''

        super(EnvPlot, self).__init__()  # Calling the __init__ of super class

        # The dimensions will be used for our canvas
        self.dim = dim
        
        # Define a 2-D observation space
        #self.observation_shape = dim
        self.observation_space = spaces.Box(low=0, high=255, shape=
                    (self.dim[0] * 20, self.dim[1] * 20, 3), dtype=np.uint8)


        # Define the action space of our gym environment
        self.action_space = spaces.Discrete(4**(agt_num))

        
        # Description of walls is stored in this array
        self.raw_occupancy = np.zeros(dim)
        self.add_walls()

        # Maintains the global state of our environment
        self.occupancy = self.raw_occupancy.copy()

        # Number of drones and boxes
        self.agt_num = agt_num
        self.box_num = box_num

        # Collection of all the agent objects
        self.agt_list = []

        for i in range(self.agt_num):
            temp_agt = Agent([4+i, 1], i)
            self.occupancy[temp_agt.pos[0], temp_agt.pos[1]] = 2
            self.agt_list.append(temp_agt)

        # Collection of all the box objects
        self.box_list = []
        self.checkbox_list = []

        for i in range(self.box_num):
            self.checkbox_list.append(1)
            temp_box = Box(self.get_unoccupied_random_cell(), i)
            self.occupancy[temp_box.pos[0], temp_box.pos[1]] = 1
            self.box_list.append(temp_box)

    def add_walls(self):
        '''Adds walls to the canvas'''

        for i in range(self.dim[0]):
            self.raw_occupancy[i, 0] = 3
            self.raw_occupancy[i, self.dim[1]-1] = 3
        for i in range(self.dim[1]):
            self.raw_occupancy[0, i] = 3
            self.raw_occupancy[self.dim[0]-1, i] = 3
            self.raw_occupancy[1, i] = 3
            if(i in range(int(self.dim[0]/2-4), int(self.dim[0]/2+4))):
                self.raw_occupancy[1, i] = 0

    def reset(self):
        '''Reset the environment'''

        self.occupancy = self.raw_occupancy.copy()

        self.agt_list = []
        self.checkbox_list = []

        for i in range(self.agt_num):
            temp_agt = Agent([4+i, 1], i)
            self.occupancy[temp_agt.pos[0], temp_agt.pos[1]] = 2
            self.agt_list.append(temp_agt)

        self.box_list = []
        for i in range(self.box_num):
            self.checkbox_list.append(1)
            temp_box = Box(self.get_unoccupied_random_cell(), i)
            self.occupancy[temp_box.pos[0], temp_box.pos[1]] = 1
            self.box_list.append(temp_box)

        #print(self.render('rgb_array').shape)
        return self.render('rgb_array')

    def step(self, action):
        '''Make all the drones move according to the given action'''

        assert self.action_space.contains(action), "Invalid Action"
        
        action_list=self.decode(action)

        done = True
        reward = -1

        for i in range(self.agt_num):

            if action_list[i] == 0:     # up
                if (self.occupancy[self.agt_list[i].pos[0] - 1][self.agt_list[i].pos[1]] != 2 and self.occupancy[self.agt_list[i].pos[0] - 1][self.agt_list[i].pos[1]] != 3):  # if can move
                    self.agt_list[i].pos[0] = self.agt_list[i].pos[0] - 1
                    self.occupancy[self.agt_list[i].pos[0] +
                                   1][self.agt_list[i].pos[1]] = 0
                    self.occupancy[self.agt_list[i].pos[0]
                                   ][self.agt_list[i].pos[1]] = 2
        
            if action_list[i] == 1:   # down
                if self.occupancy[self.agt_list[i].pos[0] + 1][self.agt_list[i].pos[1]] != 2 and self.occupancy[self.agt_list[i].pos[0] + 1][self.agt_list[i].pos[1]] != 3:  # if can move
                    self.agt_list[i].pos[0] = self.agt_list[i].pos[0] + 1
                    self.occupancy[self.agt_list[i].pos[0] -
                                   1][self.agt_list[i].pos[1]] = 0
                    self.occupancy[self.agt_list[i].pos[0]
                                   ][self.agt_list[i].pos[1]] = 2
        
            if action_list[i] == 2:   # left
                if self.occupancy[self.agt_list[i].pos[0]][self.agt_list[i].pos[1] - 1] != 2 and self.occupancy[self.agt_list[i].pos[0]][self.agt_list[i].pos[1]-1] != 3:  # if can move
                    self.agt_list[i].pos[1] = self.agt_list[i].pos[1] - 1
                    self.occupancy[self.agt_list[i].pos[0]
                                   ][self.agt_list[i].pos[1] + 1] = 0
                    self.occupancy[self.agt_list[i].pos[0]
                                   ][self.agt_list[i].pos[1]] = 2

            if action_list[i] == 3:  # right
                if self.occupancy[self.agt_list[i].pos[0]][self.agt_list[i].pos[1] + 1] != 2 and self.occupancy[self.agt_list[i].pos[0]][self.agt_list[i].pos[1]+1] != 3:  # if can move
                    self.agt_list[i].pos[1] = self.agt_list[i].pos[1] + 1
                    self.occupancy[self.agt_list[i].pos[0]
                                   ][self.agt_list[i].pos[1] - 1] = 0
                    self.occupancy[self.agt_list[i].pos[0]
                                   ][self.agt_list[i].pos[1]] = 2

        # catch box
        for i in range(self.agt_num):

            if self.agt_list[i].catch_box == -1:    # agent is not carrying any box

                for k in range(len(self.box_list)):
                    if self.agt_list[i].pos[0] == self.box_list[k].pos[0] and abs(self.agt_list[i].pos[1] - self.box_list[k].pos[1]) == 0 and self.checkbox_list[k] == 1:
                        reward=reward+5
                        self.agt_list[i].catch_box = self.box_list[k].id
                        self.checkbox_list[self.box_list[k].id] = 0
                        self.occupancy[self.agt_list[i].pos[0]
                                       ][self.agt_list[i].pos[1]] = 2
                        done = False
            else:
                done = False

        # check if box is correctly delivered

        for k in range(len(self.agt_list)):

            if self.agt_list[k].pos[1] in range(int(self.dim[0]/2-4), int(self.dim[0]/2+4)) and self.agt_list[k].pos[0] == 1 and self.agt_list[k].catch_box != -1:

                reward = reward + 20
                self.agt_list[k].catch_box = -1

        for i in self.checkbox_list:

            if i:

                done = False
                break
        
        

        state = self.render('rgb_array')

        #if(done):
            #reward=reward+100
            #Changed from 10000 to 100.

        return state, reward, done, {}

    def is_box_in_list(self, id):

        for i in range(len(self.box_list)):

            if id == self.box_list[i].id:

                return True

        return False

    def get_unoccupied_random_cell(self):
        '''Returns a point on the plot which is not occupied'''

        while(True):
            x = random.randint(0, self.dim[0]-1)
            y = random.randint(0, self.dim[1]-1)
            if(self.occupancy[x][y] == 0):
                return [x, y]

    def get_observations(self):
        '''Return the occupancy matrix after reducing it considering the reduced vision of drones'''

        obs = np.full(self.dim, -1)
        
        for i in range(self.agt_num):
            
            try:
                
                for j in range(self.agt_list[i].pos[0]-2, self.agt_list[i].pos[0]+3):
                    
                    for k in range(self.agt_list[i].pos[1]-2, self.agt_list[i].pos[1]+3):
                        
                        if(j == self.agt_list[i].pos[0] and k == self.agt_list[i].pos[1] and self.agt_list[i].catch_box != -1):
                            obs[j][k] = 4
                        
                        else:
                            obs[j][k] = self.occupancy[j][k]
            except:
                continue
            
        return obs

    def decode(self,action):
        action_list=[]
        while(len(action_list)<self.agt_num):
            action_list.append(action%4);
            action/=4;
        return action_list
    
    def render(self, mode="human"):
        '''Render  the environment to the screen'''

        obs = np.ones((self.dim[0] * 20, self.dim[1] * 20, 3),np.uint8)

        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                if self.raw_occupancy[i, j] == 3:
                    cv2.rectangle(obs, (j*20, i*20),
                                  (j*20+20, i*20+20), (0, 0, 0), -1)

        for i in range(self.agt_num):
            if self.agt_list[i].catch_box == -1:
                cv2.rectangle(obs, (self.agt_list[i].pos[1] * 20, self.agt_list[i].pos[0] * 20), (
                    self.agt_list[i].pos[1] * 20 + 20, self.agt_list[i].pos[0] * 20 + 20), (0, 255, 0), -1)
            else:
                cv2.rectangle(obs, (self.agt_list[i].pos[1] * 20, self.agt_list[i].pos[0] * 20), (
                    self.agt_list[i].pos[1] * 20 + 20, self.agt_list[i].pos[0] * 20 + 20), (0, 255, 255), -1)

        for i in range(len(self.box_list)):
            if self.checkbox_list[i]:
                cv2.rectangle(obs, (self.box_list[i].pos[1] * 20, self.box_list[i].pos[0] * 20),
                              (self.box_list[i].pos[1] * 20 + 20, self.box_list[i].pos[0] * 20 + 20), (0, 0, 255), -1)
        
        if mode == "human":
            cv2.imshow('image', obs.astype(np.float32))
            cv2.waitKey(50)

        elif mode == "rgb_array":
            return obs
    
    def close(self):
        cv2.destroyAllWindows()