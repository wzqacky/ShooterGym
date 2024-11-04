from envs.param import *

from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt 
import numpy as np

def translate_state(state): # translating the state of the game 
    state_num = 0
    for i in range(3, 3+2*N_OBSERVATIONS-1, 2):
        state_num += pow(2, (i-3)//2) * state[i]

    return state_num

def label_with_episode(frame, episode): # plotting the image 
    im = Image.fromarray(frame)
    
    drawer = ImageDraw.Draw(im)
    drawer.text((im.size[0]/20,im.size[1]/18), f'Episode: {episode+1}', fill=(0,0,0))
    return im

def plotLearning(x, scores, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax.plot(x, running_avg, color="C1")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Score')
    #ax2.xaxis.set_label_position('top')
    #ax2.tick_params(axis='x', colors="C1")
    ax.tick_params(axis='y')

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
