# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 21:16:12 2021

@author: priya
"""
import heapq 
#https://www.bogotobogo.com/python/python_Dijkstras_Shortest_Path_Algorithm.php
#https://bradfieldcs.com/algos/graphs/dijkstras-algorithm/#:~:text=Dijkstra's%20algorithm%20uses%20a%20priority,of%20vertices%20sorted%20by%20distance.

class Edge:
    def __init__(self, src, dst, cost):
        self.src = src
        self.dst = dst 
        self.cost = cost 
        
class Vertex:
    def __init__(self, id):
        self.id = id
        self.cost = float('inf')
        #self.visited = False 
        self.previous = None     
        
    def getId(self):
        return self.id
    
    def setCost(self, val):
        self.cost = val 
        #print("Updated cost of vertex: {} to {}".format(self.id, val))
    
    def getCost(self):
        return self.cost
        
    def setVisited(self, val):
        self.visited = val 
        #print("Updated visited of vertex: {} to {}".format(self.id, val))
        
    def getVisited(self):
        return self.visited
    
    def setPrevious(self, val):
        self.previous = val
        #print("Updated prev of vertex: {} to {}".format(self.id, val))
        
    def getPrevious(self):
        return self.previous
        
 
        

class Graph:
    def __init__(self, mode, costgrid):
        self.maxX, self.maxY = costgrid.shape
        self.gamemode = mode         
        self.vertices = [Vertex((x,y)) for x in range(0,self.maxX) for y in range(0,self.maxY)]
        self.edges = self.buildEdges(costgrid)
        #[Edge(e) for e in edges]
        

    #given a vertex, returns a list of all connected vertices
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
        
    #builds a list of all edges in a graph     
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
        
    #gets vertex from graph when passed an id     
    def getVertex(self, id):
        for v in self.vertices:
            if v.id == id:
                return v
        return None 
    
    #returns cost of edge between two nodes if one exists
    def getEdgeCost(self, src, dst):
        for e in self.edges:
            if e.src == src and e.dst == dst:
                return e.cost
        return None 
    
    #traverses back from destination to source to find path 
    def getPath(self, source, destination):
        path = [destination]
        current = destination 
        while 1:
            
            prev = self.getVertex(current).getPrevious()
            path.append(prev)
            
            if prev == source:
                #print("Path: ", path)
                return path 
            
            current = prev 
            
            
    def shortestPath(self, source, destination):
        
        #set source cost to 0 
        self.getVertex(source).setCost(0)

        #initialize queue
        pQueue = [(0, source)]
        
                
        while len(pQueue) > 0: #(unvisited):
            
            #get vertex with least cost  
            currentCost, currentNode = heapq.heappop(pQueue)
                        
            #check if the distance to node is already smaller than current 
            #if so, do nothing 
            if currentCost > self.getVertex(currentNode).getCost():
                continue 
            
            
            #get connected vertices for current vertex
            adjVertices = self.getConnectedVertices(currentNode[0], currentNode[1])
            #print(adjVertices)
            
            
            for v in adjVertices:
                #calculate cost to vertex 
                newCost = currentCost + self.getEdgeCost(currentNode, v)
               # print("Current node: {} with cost: {}, to node: {} with new cost: {}".format(current, cost, v, newCost))
                
                #update adjascent vertices cost if lower than current cost 
                #update previous
                #add to queue
                if newCost < self.getVertex(v).getCost():
                    self.getVertex(v).setCost(newCost)
                    self.getVertex(v).setPrevious(currentNode)
                    heapq.heappush(pQueue, (newCost, v))
                
            
            
        #get path, cost   
        path = self.getPath(source, destination)
        cost = self.getVertex(destination).getCost()
        
        return path, cost 
            







 
       

    
    
