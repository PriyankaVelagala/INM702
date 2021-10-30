# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 16:35:44 2021

@author: priya

"""


import math 
import random 
import numpy as np
import matplotlib.pyplot as plt 

from Grid import Grid
from helper_functions import *
from Graph import *
from random import seed 
from random import randint 


np.random.seed(45)

MAX_COST = 100
         
#greedy, mode 2, with random seed = 45 - runs into a corner

def main():
    print("Hello!")
    
    height = 5 #getGridSize("height")
    width = 5 #getGridSize("width")
    mode = 2 #1 #getGameMode()
    
    print("\nYou choose game mode {}, your grid size is {} x {}".format(mode, height, width))
    costGrid = np.random.randint(0, MAX_COST, (height, width))
    
    myBoard = Grid(height, width, costGrid)
    myBoard.drawGrid()
    #print(costGrid.shape)
    #path = getShortestPath("greedy", mode, costGrid, myBoard)
    path = getShortestPath("dijkstras", mode, costGrid, myBoard)
    myBoard.drawPath(path)
     
   
if __name__ == "__main__":
    main()