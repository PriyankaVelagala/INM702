# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 17:05:02 2021

@author: priya
"""

#Helper function get user input for grid dimensions
def getGridSize(side):
    while 1:
        size = input("Enter grid {}:".format(side))
        try: 
            size = int(size)
        except ValueError:
            print("{} is not a valid input, please enter an integer".format(size))
        else:
            return size

#Helper function to get user input for game mode 
def getGameMode():
    print("\nTwo game modes available: \n [1] The time spent on a cell is the number on this cell \n [2] The time spent on a cell is the absolute of the difference between the previous cell the agent was on and the current cell it is on")
    while 1:
        mode = input("Choose game mode (1 or 2): ")
        try:
            mode = int(mode)
            if mode not in (1,2):
                print("Game mode {} does not exist".format(mode))
                return getGameMode() 
            else:
                return mode
        except ValueError:
            print("{} is not a valid input".format(mode))
            
            
def djikstraPath(mode, costgrid): 
    return 1


def getLegalNodes(currentX, currentY, maxX, maxY):
    nodes = []
    #check bottom?
    if currentX + 1 < maxX: nodes.append([currentY, currentX+1])
    #check  up?
    if currentX - 1 >= 0: nodes.append([currentY, currentX-1])
    #check right
    if currentY + 1 < maxY: nodes.append([currentY+1, currentX])
    #check left 
    if currentY - 1 >= 0: nodes.append([currentY-1, currentX])
    
    return(nodes)           
        
    
"""    
def greedyPath(mode, costgrid, board):
    maxX, maxY = costgrid.shape

    pathFound = False
    
    if mode == 1:
        path = [[0,0]] 
        totalcost = 0 
        currentX = 0
        currentY = 0
        
        while pathFound == False:
            toVisit = []
            allLegalNodes = getLegalNodes(currentX, currentY, maxX, maxY)
            
            for node in allLegalNodes:
                y,x = node 
                inpath = False
                for point in path:
                    if point[0] == x and point[1] == y:
                        inpath = True
                        break
                if inpath == False: toVisit.append([x,y])
                #if node not in path: toVisit.append(node)

            print("Nodes to visit: ",  [costgrid[x[0]][x[1]] for x in toVisit])
           
            cost = float('inf')
            for x,y in toVisit:
                if x == maxX-1 and y == maxY-1: 
                    cost = costgrid[x][y]
                    pathFound = True 
                    nextnode = [x,y]
                    break
                if costgrid[x][y] < cost:
                    cost = costgrid[x][y]
                    nextnode = [x,y]

            totalcost += cost 
            path.append(nextnode)
            currentX = nextnode[0]
            currentY = nextnode[1]
            board.drawPath(path)
            
    elif mode == 2:
        path = [[0,0]] 
        totalcost = 0 
        currentX = 0
        currentY = 0
        currentcost = costgrid[0][0]
        
        while pathFound == False:
            toVisit = []
            allLegalNodes = getLegalNodes(currentX, currentY, maxX, maxY)
            
            for node in allLegalNodes:
                y,x = node 
                inpath = False
                for point in path:
                    if point[0] == x and point[1] == y:
                        inpath = True
                        break
                if inpath == False: toVisit.append([x,y])
                #if node not in path: toVisit.append(node)

            print("Nodes to visit: ",  [costgrid[x[0]][x[1]] for x in toVisit])
           
            cost = float('inf')
            for x,y in toVisit:
                currentcost = costgrid[path[-1][0]][path[-1][1]]
                
                if x == maxX-1 and y == maxY-1: 
                    cost = costgrid[x][y]
                    pathFound = True 
                    nextnode = [x,y]
                    break
                if abs(costgrid[x][y]-currentcost) < cost:
                    cost = abs(costgrid[x][y]-currentcost)
                    nextnode = [x,y]

            totalcost += cost 
            path.append(nextnode)
            currentX = nextnode[0]
            currentY = nextnode[1]
            
            #print("path: ", [costgrid[x[0]][x[1]] for x in path])
            #print("total cost: ", totalcost)
            
            board.drawPath(path)

        return path
            
 """           
        
def greedyPath(mode, costgrid, board):
    maxX, maxY = costgrid.shape

    pathFound = False
    path = [[0,0]] 
    totalcost = 0 
    currentX = 0
    currentY = 0
    currentcost = costgrid[0][0]

        
    while pathFound == False:
        toVisit = []
        allLegalNodes = getLegalNodes(currentX, currentY, maxX, maxY)
            
        for node in allLegalNodes:
            y,x = node 
            inpath = False
            for point in path:
                if point[0] == x and point[1] == y:
                    inpath = True
                    break
            if inpath == False: toVisit.append([x,y])
            #if node not in path: toVisit.append(node)

        print("Nodes to visit: ",  [costgrid[x[0]][x[1]] for x in toVisit])
       
        cost = float('inf')
        for x,y in toVisit:
            currentcost = costgrid[path[-1][0]][path[-1][1]]
            if x == maxX-1 and y == maxY-1: 
                cost = costgrid[x][y]
                pathFound = True 
                nextnode = [x,y]
                break
            if mode == 1: 
                if costgrid[x][y] < cost:
                    cost = costgrid[x][y]
                    nextnode = [x,y]
            elif mode == 2:
                if abs(costgrid[x][y]-currentcost) < cost:
                    cost = abs(costgrid[x][y]-currentcost)
                    nextnode = [x,y]
        totalcost += cost 
        path.append(nextnode)
        currentX = nextnode[0]
        currentY = nextnode[1]
        board.drawPath(path)
    return path
    

def getShortestPath(algo, mode, costgrid, board):
    if algo == "greedy":
        path = greedyPath(mode, costgrid, board)
    else:
        path = djikstraPath(mode, costgrid) 
    return path
        