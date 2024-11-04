import math
from typing import Tuple

from gym.envs.classic_control import rendering
from gym.envs.classic_control.rendering import Geom

from envs.param import *


class Entity:
    def __init__(self, pos, type):
        self.x = pos[0]
        self.y = pos[1]
        self.type = type
        self.rect = ENTITIES.get(type).get('rect')
        self.velocity = ENTITIES.get(type).get('initial_velocity') 
        self.trans = rendering.Transform(translation=pos)
        self.shape = self.build_shape()

    def build_shape(self) -> Geom:
        img = rendering.Image(ENTITIES.get(self.type).get('shape'), self.rect[0], self.rect[1])
        img.set_color(1., 1., 1.)
        img.add_attr(self.trans)
        return img

class Player(Entity):
    def __init__(self, pos):
        super().__init__(pos, 'player')
        self.direction = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]

    def shoot(self, choice):
        return Bullet((self.x, self.y), self.direction[choice])

    def advance(self, choice):
        self.x += self.direction[choice][0] * self.velocity
        self.y += self.direction[choice][1] * self.velocity
        # to check if the character go out of bound 
        if self.x < 0: self.x = 0 
        if self.x > 600: self.x = 600
        if self.y < 0: self.y = 0 
        if self.y > 600: self.y = 600
        self.trans.set_translation(self.x, self.y)
        
class Bullet(Entity):
    def __init__(self, pos, direction):
        super().__init__(pos, 'bullet')
        self.direction = direction
    
    def advance(self):
        self.x += self.direction[0] * self.velocity
        self.y += self.direction[1] * self.velocity
        self.trans.set_translation(self.x, self.y)

    """
    # checks collision between bullet and enemy
    def collide_with_enemy(self): 
        if pygame.sprite.spritecollide(self, enemy_group, True):
            score += 1
            edeath_sound.play()
      

    def update(self):
        # remove bullets outside the screen 
        if 0 < self.rect.x < 600:
            self.rect.x += self.x_speed 
        else:
            self.kill()
        if 0 < self.rect.y < 600:
            self.rect.y += self.y_speed 
        else:
            self.kill()
        self.collide_with_enemy() 
    """ 

class Enemy(Entity):
    def __init__(self, pos):
        super().__init__(pos, 'enemy')

    def advance(self, target_x, target_y):
        #compute directional normal vector (dx, dy)
        dx, dy = target_x - self.x, target_y- self.y
        abs_dist = math.hypot(abs(dx), abs(dy))
        self.isleft = 1 if dx < 0 else 0
        if abs_dist > 0:
            self.x += (dx/abs_dist) * self.velocity
            self.y += (dy/abs_dist) * self.velocity
        self.trans.set_translation(self.x, self.y)

    """ 
    def animation(self):
        self.image = self.walk[self.isleft]
    
    def update(self):
        self.enemy_movement()
        self.animation()
    """

def entity_intersection(e1, e2):
    """
    Check if two entities are intersecting.
    It uses a circle-circle hitbox intersection.

    :param e1: The first entity
    :param e2: The second entity
    :return: True if the entity are intersecting
    """
    d1 = math.fabs(e1.x - e2.x) # fabs: float absolute 
    d2 = math.fabs(e1.y - e2.y)
    e1.radius = math.sqrt(pow(e1.rect[0], 2) + pow(e1.rect[1], 2))
    e2.radius = math.sqrt(pow(e2.rect[0], 2) + pow(e2.rect[1], 2))
    return (d1 < e1.radius or d1 < e2.radius) and (d2 < e1.radius or d2 < e2.radius)


def line_entity_intersection(p1, p2, e):
    """
    Check if the line defined by two points intersect with the entity

    :param p1: The first point
    :param p2: The second point
    :param e: The entity
    :return: A tuple with the entity type and its distance from the original point, else (0.0, 0.0)
    """
    x1, y1 = p1
    x2, y2 = p2
    r = math.sqrt(pow(e.rect[0], 2) + pow(e.rect[1], 2))
    # entity center
    x0 = e.x
    y0 = e.y

    # compute distance point-line, proof: https://www.youtube.com/watch?v=9AS9IraO_08
    dist = math.fabs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / math.sqrt(
        math.pow(y2 - y1, 2) + math.pow(x2 - x1, 2))
    if dist <= r:
        # make sure we're in the same quadrant
        if included(x1, x2, x0) and included(y1, y2, y0):
            return ENTITIES.get(e.type).get('value'), distance(x1, y1, e)
    return 0, 0


def border_distance(x, y, theta):
    """
    Computes the closest distance from a point (x,y) to the window borders given the angle theta (in radians)

    :param x: The X point coordinate
    :param y:  The Y point coordinate
    :param theta: The angle (in radians)
    :return: The closest distance to the window borders
    """
    x1 = x + max(SCREEN_HEIGHT, SCREEN_WIDTH) * math.cos(theta)
    y1 = y + max(SCREEN_HEIGHT, SCREEN_WIDTH) * math.sin(theta)
    # intersection with borders
    left_x, left_y = line_line_intersection(x, y, x1, y1, 0, 0, 0, SCREEN_HEIGHT)
    if included(x, x1, left_x) and included(y, y1, left_y):
        return math.sqrt(math.pow(x - left_x, 2) + math.pow(y - left_y, 2))
    right_x, right_y = line_line_intersection(x, y, x1, y1, SCREEN_WIDTH, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
    if included(x, x1, right_x) and included(y, y1, right_y):
        return math.sqrt(math.pow(x - right_x, 2) + math.pow(y - right_y, 2))
    down_x, down_y = line_line_intersection(x, y, x1, y1, 0, 0, SCREEN_WIDTH, 0)
    if included(x, x1, down_x) and included(y, y1, down_y):
        return math.sqrt(math.pow(x - down_x, 2) + math.pow(y - down_y, 2))
    up_x, up_y = line_line_intersection(x, y, x1, y1, 0, SCREEN_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT)
    if included(x, x1, up_x) and included(y, y1, up_y):
        return math.sqrt(math.pow(x - up_x, 2) + math.pow(y - up_y, 2))
    # this should never happen
    return 0


def line_line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Compute the line to line intersection

    (X1, Y1), (X2, Y2) belongs to L1
    (X3, Y3), (X4, Y4) belongs to L2

    :param x1: The coordinate X1
    :param y1: The coordinate Y1
    :param x2: The coordinate X2
    :param y2: The coordinate Y2
    :param x3: The coordinate X3
    :param y3: The coordinate Y3
    :param x4: The coordinate X4
    :param y4: The coordinate Y4
    :return: The intersection point
    """
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    # failsafe if the lines are parallel
    if d == 0:
        return math.nan, math.nan
    # compute intersection point
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
    return px, py


def distance(x, y, e):
    """
    Compute the distance between a point and an entity's center

    :param x: The point's X coordinate
    :param y: The point's Y coordinate
    :param e: The entity
    :return: The distance between point and entity's center
    """
    return math.sqrt(math.pow(e.x - x, 2) + math.pow(e.y - y, 2))


def included(a, b, v):
    """
    Check that v is a <= v <= b

    :param a: The lower bound
    :param b: The upper bound
    :param v: The value to check
    :return: The inclusion condition
    """
    return a <= v <= b if a <= b else b <= v <= a