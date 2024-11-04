import math

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
DIAG = math.sqrt(math.pow(SCREEN_WIDTH, 2) + math.pow(SCREEN_HEIGHT, 2))
N_OBSERVATIONS = 8
BORDER_VALUE = 0

# -- Entities properties --
ENTITIES = {
    'player': {
        'rect': (45, 60), # the actual width and height of the collding block 
        'initial_velocity': 3,
        'shape': 'envs/assets/sold_right.png'
    },
    'bullet': {
        'rect': (32, 32),
        'initial_velocity': 6,
        'shape': 'envs/assets/bullet.png'
    },
    'enemy': {
        'rect': (34, 49),
        'initial_velocity': 2,
        'shape': 'envs/assets/enemy_left_1.png',
        'value': 1
    }
}

# rewards
KILLED_ENEMY = 200
DIED = -400