
from PIL import Image
import random
import numpy
import pdb

from PIL import Image

import array
import logging

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
 

class Set(object):
    def __init__(self, org_image, gt_image):
        self.org_image = org_image
        self.gt_image = gt_image
        self.org_image_map = []
        self.gt_image_map = []
        self.width = 0
        self.height = 0
        self.minority_pixel_loc = []
        self.majority_pixel_loc = []
        self.sminf = []
        self.sbmaj = []
        self.simin = []
        self.k1 = 8
        self.k2 = 3
        self.k3 = 3


    def openImages(self):
        self.gt_image_map = Image.open(self.gt_image).load()
        self.org_image_map = Image.open(self.org_image).load()

        self.width, self.height = Image.open(self.gt_image).size

    def findSminf(self, i, j):

        #i associates with width
        #j associates with height

        count_majority = 0

        for row in range(3):
            #to traverse throug -1 to +1
            row_x = j  + (row-1)

            for colum in range(3):
                
                col_x = i + (colum -1)

                #check the boundary of image width and height
                if col_x < self.width and row_x<self.height and col_x>0 and row_x>0:
                    if self.calcDistance(self.gt_image_map[col_x,row_x], self.gt_image_map[41,22]) != 0.0: 
                        count_majority = count_majority+1


        print count_majority

        #if all the 8 neghbours are majority remove, else add

        # here we are only adding the sminf
        if count_majority < self.k2:
            self.sminf.append((i,j))


    def findSbmaj(self, i , j):
        #i associates with width
        #j associates with height

        count_majority = 0

        for row in range(3):
            #to traverse throug -1 to +1
            row_x = j  + (row-1)

            for colum in range(3):
                
                col_x = i + (colum -1)

                #check the boundary of image width and height
                if col_x < self.width and row_x<self.height and col_x>0 and row_x>0:
                    #if it is in majority
                    if self.calcDistance(self.gt_image_map[col_x,row_x], self.gt_image_map[41,22]) != 0.0: 
                        count_majority = count_majority+1


        print count_majority

        #if all the 8 neghbours are majority remove, else add

        # consider count of majority, if it is  <= 3
        #add all the majority set to sbmaj
        if count_majority <= self.k2:

            for row in range(3):
                row_x = j  + (row-1)

                for colum in range(3):
                
                    col_x = i + (colum -1)

                    #check the boundary of image width and height
                    if col_x < self.width and row_x<self.height and col_x>0 and row_x>0:
                        #if it is in majority
                        if self.calcDistance(self.gt_image_map[col_x,row_x], self.gt_image_map[41,22]) != 0.0: 
                            # add the majority set
                            self.sbmaj.append((col_x, row_x))

    def findSimin(self):
        #iterate though sbmaj set
        for idx in self.sbmaj:
            i = idx[0]
            j = idx[1]

            count_minority = 0

            for row in range(3):
                row_x = j  + (row-1)

                for colum in range(3):
                
                    col_x = i + (colum -1)

                    #check the boundary of image width and height
                    if col_x < self.width and row_x<self.height and col_x>0 and row_x>0:
                        #if it is in minority
                        if self.calcDistance(self.gt_image_map[col_x,row_x], self.gt_image_map[41,22]) == 0.0: 
                            count_minority = count_minority + 1

            # add to simin 
            if count_minority<=self.k3:
                for row in range(3):
                    row_x = j  + (row-1)

                    for colum in range(3):
                    
                        col_x = i + (colum -1)

                        #check the boundary of image width and height
                        if col_x < self.width and row_x<self.height and col_x>0 and row_x>0:
                            #if it is minority
                            if self.calcDistance(self.gt_image_map[col_x,row_x], self.gt_image_map[41,22]) == 0.0: 
                                self.simin.append((col_x, row_x))

    def removeDuplicates(self):
        cleanlist = []
        [cleanlist.append(x) for x in self.simin if x not in cleanlist]
        self.simin = []
        self.simin = cleanlist

        cleanlist = []
        [cleanlist.append(x) for x in self.sbmaj if x not in cleanlist]
        self.sbmaj = []
        self.sbmaj = cleanlist



    def seperate_classes(self):
        for i in range(self.width):
            for j in range(self.height):
               
                #match with exact white pixel
                if self.calcDistance(self.gt_image_map[i,j], self.gt_image_map[41,22]) == 0.0: 
                    self.minority_pixel_loc.append((i,j))

                    self.findSminf(i, j)
                    self.findSbmaj(i, j)
                else:
                    self.majority_pixel_loc.append((i,j))

        self.findSimin()

        print len(self.sminf), len(self.minority_pixel_loc), len(self.sbmaj), len(self.simin)

        self.removeDuplicates()


    def calcDistance(self, a, b):
        dis1 = pow((a[0]-b[0]),2.0)
        dis2 = pow((a[1]-b[1]),2.0)
        dis3 = pow((a[2]-b[2]),2.0)
        dis4 = pow((a[3]-b[3]),2.0)

        sumation = dis1 + dis2 + dis3 + dis4
        result = numpy.sqrt(sumation)
        return result

    def getSets(self):
        return self.majority_pixel_loc, self.minority_pixel_loc, self.sminf, self.sbmaj, self.simin


    






if __name__ == "__main__":
    s= Set("T3.png", "GT_T3.png")
    s.openImages()
    s.seperate_classes()
    majority, minority, sminf, sbmaj, simin = s.getSets()