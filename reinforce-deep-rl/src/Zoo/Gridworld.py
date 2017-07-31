import numpy as np
import random
import itertools
from scipy.misc import imresize
from PIL import Image

import gym
from gym import error, spaces, utils
from gym.utils import seeding

class Gridworld(gym.Env):
    class GridObj:
        def __init__(self, coordinates, size, intensity, channel, reward, name):
            self.x         = coordinates[0]
            self.y         = coordinates[1]
            self.size      = size
            self.intensity = intensity
            self.channel   = channel
            self.reward    = reward
            self.name      = name

    def __init__(self, partial, sizeX, sizeY=None):
        self.sizeX   = sizeX
        self.sizeY   = sizeX if sizeY == None else sizeY
        self.actions = 4
        self.objects = []
        self.partial = partial
        self.state   = None
        self.hero    = None

    def _step(self, action):
        # 0 - up, 1 - down, 2 - left, 3 - right

        hero         = self.objects[0]
        heroX, heroY = hero.x, hero.y
        penalize     = 0.0

        action_up    = lambda a: action == 0
        action_down  = lambda a: action == 1
        action_left  = lambda a: action == 2
        action_right = lambda a: action == 3

        can_up       = lambda hero: hero.y >= 1
        can_down     = lambda hero: hero.y <= (self.sizeY - 2)
        can_left     = lambda hero: hero.x >= 1
        can_right    = lambda hero: hero.x <= (self.sizeX - 2)

        def move_up(hero): hero.y -= 1
        def move_down(hero): hero.y += 1
        def move_left(hero): hero.x -= 1
        def move_right(hero): hero.x += 1

        hero_unmoved = lambda hero: hero.x == heroX and hero.y == heroY


        if action_up(action) and can_up(hero):
            move_up(hero)

        elif action_down(action) and can_down(hero):
            move_down(hero)

        elif action_left(action) and can_left(hero):
            move_left(hero)

        elif action_right(action) and can_right(hero):
            move_right(hero)

        elif hero_unmoved(hero):
            penalize = 0.0

        else:
            raise Exception("non-exhaustive stepping function")

        self.objects[0] = hero  # i don't think this needs to happen as this is mutable

        penalty = penalize
        reward, done = self.checkGoal()
        state = self._render()

        if reward == None:
            print(done)
            print(reward)
            print(penalty)
            print('error, reward is none')
            reward = 0
        return state, reward+penalty, done


    def _reset(self):
        self.hero    = self.GridObj(self.newPosition(),1,1,2,None,'hero')
        self.objects = []
        self.objects.append(self.hero)
        self.objects.append(self.GridObj(self.newPosition(),1,1,1,   1,'goal'))
        self.objects.append(self.GridObj(self.newPosition(),1,1,0,  -1,'fire'))
        self.objects.append(self.GridObj(self.newPosition(),1,1,1,   1,'goal'))
        self.objects.append(self.GridObj(self.newPosition(),1,1,0,  -1,'fire'))
        self.objects.append(self.GridObj(self.newPosition(),1,1,1,   1,'goal'))
        self.objects.append(self.GridObj(self.newPosition(),1,1,1,   1,'goal'))
        state = self._render()
        self.state = state
        return state

    def _render(self, mode='human', close=False):
        a = np.ones([self.sizeY + 2,self.sizeX + 2,3])
        a[1:-1,1:-1,:] = 0

        for item in self.objects:
            itemXs = item.x + 1
            itemXf = item.x + item.size + 1

            itemYs = item.y + 1
            itemYf = item.y + item.size + 1

            a[itemYs:itemYf, itemXs:itemXf, item.channel] = item.intensity

        hero = self.find_hero()

        if self.partial == True:
            a = a[hero.y:hero.y+3, hero.x:hero.x+3, :]
        b = imresize(a[:,:,0],[84,84,1])
        c = imresize(a[:,:,1],[84,84,1])
        d = imresize(a[:,:,2],[84,84,1])
        a = np.stack([b,c,d],axis=2)
        return a

    def find_hero(self):
        for obj in self.objects:
            if obj.name == 'hero':
                return obj
        raise Exception("no hero found")
 

    def checkGoal(self):
        hero   = self.find_hero()
        others = filter(lambda obj: obj.name != 'hero', self.objects)
        reward = None
        done   = False

        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)

                if other.reward == 1:
                    self.objects.append(self.GridObj(self.newPosition(),1,1,1, 1,'goal'))
                else:
                    self.objects.append(self.GridObj(self.newPosition(),1,1,0,-1,'fire'))

                return other.reward, done

        if done == False:
            return 0.0, False

    def newPosition(self):
        iterables = [ range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x,objectA.y) not in currentPositions:
                currentPositions.append((objectA.x,objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)),replace=False)
        return points[location]

#def imresize(arr, size):
#    high=255
#    low=0
#    data = np.asarray(arr)
#    shape = (data.shape[1], data.shape[0])  # columns show up first
#    scale = float(high - low) / (data.max() - data.min())
#    bytedata = (data - data.min()) * scale + low
#    bytedata = (bytedata.clip(low, high) + 0.5).astype(np.uint8)
#    return np.array(
#            Image.frombytes('L', shape, bytedata.tostring())
#                .resize((size[1], size[0]), resample=0))

