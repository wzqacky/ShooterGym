from envs.entities import *
from envs.param import *

import gym
import numpy as np
import random
from gym import spaces, logger
from gym.utils import seeding


def init_scene(env):
    # clean existing scene
    env.player = None 
    env.enemies = []
    env.bullets = []

    # initialize the scene
    # add the player
    env.player = Player((SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))


class ShooterEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        """
        Create the environment
        """

        """
        possible action:
        0 = left 
        1 = up-left 
        2 = up
        3 = up-right
        4 = right 
        5 = bottom-right 
        6 = bottom
        7 = bottom-left 
        """

        self.action_space = spaces.Discrete(8)
        # observation space
        self.observation_space = spaces.Box(np.zeros(3 + N_OBSERVATIONS * 2),
                                            np.ones(3 + N_OBSERVATIONS * 2),
                                            dtype=np.int64)
        # current state
        self.state = None
        # incremental angle for observation 
        self.dtheta = math.radians(360 / N_OBSERVATIONS)
        # flag for end of episode
        self.done = False
        # flag for executions after end of episode
        self.steps_beyond_done = None

        # scene's entities
        self.player = None
        self.enemies = []
        self.bullets = []

        self.bullet_time = None 
        self.enemy_time = None 
        self.enemy_limit = 50

        self.reward = 0

        # renderer
        self.viewer = None

        # random seed fixing
        self.np_random = None
        self.seed(42)

        # initialize the scene
        init_scene(self)

    def seed(self, seed: int = None):
        """
        Fix the random seed for reproducibility
        :param seed: Random seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int):
        """
        Apply a single step in the environment using the given action
        :param action: The action to apply
        """
        # sanity check for the action
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # execute the action if possible
        if not self.done:
            if action is not None:
                self.player.advance(action)

            # spawning the bullets 
            if self.bullet_time is None:
                self.bullet_time = 0
                self.bullet_time = self.bullet_spawn(self.bullet_time, action)
            else:
                self.bullet_time = self.bullet_spawn(self.bullet_time, action)

            # updating the bullets 
            for bullet in self.bullets:
                bullet.advance()
                self.check_bounds(bullet, self.bullets)

            # spawning the enemies 
            if self.enemy_time is None:
                self.enemy_time = 0
                self.enemy_time = self.enemy_spawn(self.enemy_time)
            else:
                self.enemy_time = self.enemy_spawn(self.enemy_time)

            # remove enemies if hit by bullet
            for enemy in self.enemies:
                enemy.advance(self.player.x, self.player.y)
                for bullet in self.bullets:
                    if entity_intersection(enemy, bullet):
                        self.bullets.remove(bullet)
                        self.viewer.geoms.remove(bullet.shape)
                        self.enemies.remove(enemy)
                        self.viewer.geoms.remove(enemy.shape)
                        # increment reward
                        self.reward += KILLED_ENEMY
                        break  # no need to check for other bullets hitting the same enemy

            # remove the player if collided with enemy 
            for enemy in self.enemies:
                if entity_intersection(self.player, enemy):
                    self.done = True  # terminate session
                    self.viewer.geoms.remove(self.player.shape)
                    # decrease reward
                    self.reward += DIED
                    break  # no need to check for other enemies hitting the spaceship

            self.make_observations(action)
        return self.state, self.reward, self.done, {}

    def reset(self):
        """
        Reset the current scene, computing the observations
        :return: The state observations
        """
        # reset the scene
        init_scene(self)
        self.done = False
        self.reward = 0
        self.steps_beyond_done = None
        self.reset_geoms()
        # compute obs
        self.make_observations(0)
        return self.state

    def render(self, mode='human'):
        """
        Render the current state of the scene
        :param mode: The rendering mode to use
        """
        if self.viewer is None:
            self.viewer = rendering.Viewer(SCREEN_WIDTH, SCREEN_HEIGHT)
            self.reset_geoms()

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        """
        Terminate the episode
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    # -- Sugar coding functions

    def check_bounds(self, entity, arr):
        """
        Check if an entity is within the window bounds and remove it from the scene in case it is
        :param entity: The entity to check
        :param arr: The array of the entity type
        """
        if entity.x >= SCREEN_WIDTH or entity.y >= SCREEN_HEIGHT or entity.x <= 0.0 or entity.y <= 0.0:
            # remove it from the scene
            arr.remove(entity)
            # remove it from the renderer
            self.viewer.geoms.remove(entity.shape)

    def reset_geoms(self):
        """
        Reset the entities in the renderer
        """
        if self.viewer:
            self.viewer.geoms = []
            self.viewer.add_geom(self.player.shape)

    # -- Interacting with the environment --

    def make_observations(self, action):
        """
        Compute all observations, updating the state
        """
        # reset state
        self.state = np.zeros(3 + N_OBSERVATIONS * 2, dtype=np.int64)

        # player status: pos_x, pos_y, facing_direction 
        self.state[0] = self.player.x 
        self.state[1] = self.player.y 
        self.state[2] = action 

        # make observations
        alpha = 0
        for i in range(3, 3 + 2 * N_OBSERVATIONS-1, 2):
            x = self.player.x + (max(SCREEN_HEIGHT, SCREEN_WIDTH)) * math.cos(
                alpha * self.dtheta)
            y = self.player.y + (max(SCREEN_HEIGHT, SCREEN_WIDTH)) * math.sin(
                alpha * self.dtheta)

            # check for enemies
            for enemy in self.enemies:
                t, d = line_entity_intersection((self.player.x, self.player.y), (x, y), enemy)
                if t != 0: # intersect with the enemies 
                    self.state[i] = t
                    self.state[i+1] = d 

                    # debugging lines
                    if self.viewer:
                        line = self.viewer.draw_line((self.player.x, self.player.y), (enemy.x, enemy.y))
                        line.set_color(1., 0., 0.)
                        self.viewer.add_onetime(line)

            # else, set border distance (as nothing exists in that angle)
            if self.state[i] == 0:
                self.state[i] = BORDER_VALUE
                self.state[i + 1] = border_distance(self.player.x, self.player.y, alpha * self.dtheta)
            alpha += 1

    def bullet_spawn(self, time_bullet, action):
        time_bullet += 1 / 15
        if int(time_bullet) >= 1:
            bul = self.player.shoot(action)
            self.viewer.add_geom(bul.shape)
            self.bullets.append(bul)
            time_bullet = 0
        return time_bullet
    
    def enemy_spawn(self, time_enemy):
        time_enemy += 1 / 50
        if int(time_enemy) >= 1 and len(self.enemies) < self.enemy_limit:
            ene = Enemy(random.choice([(0, 0), (0, SCREEN_HEIGHT/2), (0, SCREEN_HEIGHT), (SCREEN_WIDTH/2, 0), (SCREEN_WIDTH/2, SCREEN_HEIGHT), 
                                          (SCREEN_WIDTH, 0), (SCREEN_WIDTH, SCREEN_HEIGHT/2), (SCREEN_WIDTH, SCREEN_HEIGHT)]))
            self.viewer.add_geom(ene.shape)
            self.enemies.append(ene)
            time_enemy = 0
        return time_enemy
