# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 21:16:12 2021

@author: priya
"""

#https://www.bogotobogo.com/python/python_Dijkstras_Shortest_Path_Algorithm.php

class Edge:
    def __init__(self, src, dst, cost):
        self.src = src
        self.dst = dst 
        self.cost = cost 
        
class Vertex:
    def __init__(self, id):
        self.id = id
        self.cost = float('inf')
        self.visited = False 
        self.previous = None     
        
    def getId(self):
        return self.id
    
    def setCost(self, val):
        self.cost = val 
        print("Updated cost of vertex: {} to {}".format(self.id, val))
    
    def getCost(self):
        return self.cost
        
    def setVisited(self, val):
        self.visited = val 
        print("Updated visited of vertex: {} to {}".format(self.id, val))
        
    def getVisited(self):
        return self.visited
    
    def setPrevious(self, val):
        self.previous = val
        print("Updated prev of vertex: {} to {}".format(self.id, val))
        
    def getPrevious(self):
        return self.previous
        
 
        

class Graph:
    def __init__(self, mode, costgrid):
        self.maxX, self.maxY = costgrid.shape
        self.gamemode = mode         
        self.vertices = [Vertex((x,y)) for x in range(0,self.maxX) for y in range(0,self.maxY)]
        self.edges = self.buildEdges(costgrid)
        #[Edge(e) for e in edges]
        

     
    def getConnectedVertices(self, currentX, currentY):
        maxX = self.maxX
        maxY = self.maxY
        vertices = []
        #check bottom?
        if currentX + 1 < maxX: vertices.append((currentX+1, currentY))
        #check  up?
        if currentX - 1 >= 0: vertices.append((currentX-1, currentY))
        #check right
        if currentY + 1 < maxY: vertices.append((currentX, currentY+1))
        #check left 
        if currentY - 1 >= 0: vertices.append((currentX, currentY-1))
        return(vertices)  
        
        
    def buildEdges(self, costgrid):
        edges = [] 
        maxX, maxY = costgrid.shape 
        
        for x in range(0,maxX):
            for y in range(0,maxY):
                srcCost = costgrid[x][y]
                target = self.getConnectedVertices(x, y)
                for node in target:
                    i,j = node
                    cost = costgrid[i][j]
                    if self.gamemode == 1: 
                        edges.append(Edge((x,y), (i,j), cost ))
                    else: 
                        edges.append(Edge((x,y), (i,j), abs(cost-srcCost) ))
                #print("current node: ({},{}) , connected nodes: {}".format(i, j, target))
            
        return edges
        
        
    def getVertex(self, id):
        for v in self.vertices:
            if v.id == id:
                return v
        return None 
    
    def getEdgeCost(self, src, dst):
        for e in self.edges:
            if e.src == src and e.dst == dst:
                return e.cost
        return None 
    
    def getPath(self, source, destination):
        path = [destination]
        current = destination 
        while 1:
            
            prev = self.getVertex(current).getPrevious()
            path.append(prev)
            
            if prev == source:
                print("Path: ", path)
                return path 
            
            current = prev 
            
            
            
        
        
        
    def shortestPath(self, source, destination):
        
        #set source cost to 0 
        self.getVertex(source).setCost(0)
        current = source 
        
        unvisited = [(x,y) for x in range(0,self.maxX) for y in range(0,self.maxY)]
        
        
        i = 0 
        
        while (unvisited):
            
            cost = self.getVertex(current).getCost()
            
            self.getVertex(current).setVisited(True)
            
            #get connected vertices for current vertex
            adjVertices = self.getConnectedVertices(current[0], current[1])
            print(adjVertices)
            
            for v in adjVertices:
                #check if vertex visited
                if self.getVertex(v).getVisited():
                    continue 
                
                #if not, calculate cost to vertex 
                newCost = cost + self.getEdgeCost(current, v)
                print("Current node: {} with cost: {}, to node: {} with new cost: {}".format(current, cost, v, newCost))
                
                #update adjascent vertices cost if lower than edge cost 
                if newCost < self.getVertex(v).getCost():
                    self.getVertex(v).setCost(newCost)
                    self.getVertex(v).setPrevious(current)
                   
                
            
            #relax current edge 
            unvisited.remove(current)
            
            #find next vertex 
            nextVertex = None 
            minCost = float('inf')
            for v in self.vertices:
                if self.getVertex(v.getId()).getCost() < minCost and self.getVertex(v.getId()).getVisited() == False:
                    nextVertex = v.getId()
                    minCost = self.getVertex(v.getId()).getCost()
            print("Next vertex to visit: {}".format(nextVertex))
                
            current = nextVertex
            
            print("Remaining nodes to visit: ", len(unvisited))
            
          
        path = self.getPath(source, destination)
        
        return path 
                
                
            
            
   






 
       

    
    
