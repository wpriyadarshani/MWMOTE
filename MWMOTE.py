
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
        self.N_min = {}
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
                #to add to Nmin
                neightbour = []

                for row in range(3):
                    row_x = j  + (row-1)

                    for colum in range(3):
                    
                        col_x = i + (colum -1)

                        #check the boundary of image width and height
                        if col_x < self.width and row_x<self.height and col_x>0 and row_x>0:
                            #if it is minority
                            if self.calcDistance(self.gt_image_map[col_x,row_x], self.gt_image_map[41,22]) == 0.0: 
                                self.simin.append((col_x, row_x))
                                neightbour.append((col_x, row_x))

                #update the N_min
                self.N_min[idx] = neightbour

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
        return self.majority_pixel_loc, self.minority_pixel_loc, self.sminf, self.sbmaj, self.simin, self.N_min, self.org_image_map


class Weight(object):
    def __init__(self, N_min, sbmaj, simin, org_image_map):
        self.N_min = N_min
        self.sbmaj = sbmaj
        self.simin = simin
        self.org_image_map = org_image_map
        self.c_max = 2.0
        self.c_th = 5.0
       
        self.I_w = {}
        self.S_w = {}
        self.S_p = {}


    def generateWeights(self):

        closeness_factor_dic = {}

        for y in self.sbmaj:
            sum_C_f = 0.
            for x in self.simin:
              # closeness_factor
              #check given sbmaj is in the N_min
                if y in self.N_min:
                    if x not in  self.N_min[y]:
                        closeness_factor = 0.
                    else:
                        #calculate the normalized eucledian distance
                        distance = self.calcEucledianDistance(x, y)
                        normalized_distance =  distance/255.0

                        if normalized_distance != 0:
                            #min - closeness factor lied between the [0, self.c_max]
                            closeness_factor = min(self.c_th, (1 / normalized_distance)) / self.c_th * self.c_max
                            closeness_factor_dic[(y,x)] = closeness_factor
                            sum_C_f +=closeness_factor
                        else:

                            #if pixel values are equal remove the normalized distance and calculate rest

                            closeness_factor = min(self.c_th, (1 )) / self.c_th * self.c_max
                            closeness_factor_dic[(y,x)] = closeness_factor
                            sum_C_f +=closeness_factor
                            # print closeness_factor

        ############dense factor calcultion############################



            for x in self.simin:
                key = (y,x)
                if key in closeness_factor_dic:
                    closeness_factor = closeness_factor_dic[(y, x)]
                    densityFactor = closeness_factor/ sum_C_f

                    #weight =  closenes factor * density factor
                    d = densityFactor * closeness_factor

                    #add the weight
                    self.I_w[(y,x)] = d

    def generateSWSP(self):

        for x in self.simin:
            sw = 0.0
            for y in self.sbmaj:
                key = (y,x)
                if key in self.I_w:
                    sw+=self.I_w[(y, x)]
                    
            self.S_w[x] = sw

        
        WeightSum = math.fsum(self.S_w.values())
        for x in self.S_w:
            self.S_p[x] = float(self.S_w[x])/WeightSum

    def calcEucledianDistance(self, min_loc, maj_loc):
        #get the pixel value from the org_image_map
        maj_pixel = self.org_image_map[maj_loc[0], maj_loc[1]]
        min_pixel = self.org_image_map[min_loc[0], min_loc[1]]

        dis1 = pow((maj_pixel[0]-min_pixel[0]),2.0)
        dis2 = pow((maj_pixel[1]-min_pixel[1]),2.0)
        dis3 = pow((maj_pixel[2]-min_pixel[2]),2.0)
        dis4 = pow((maj_pixel[3]-min_pixel[3]),2.0)

        sumation = dis1 + dis2 + dis3 + dis4
        result = numpy.sqrt(sumation)
        return result

    def getSets(self):
        return self.I_w, self.S_w, self.S_p


if __name__ == "__main__":
    s= Set("T3.png", "GT_T3.png")
    s.openImages()
    s.seperate_classes()
    majority, minority, sminf, sbmaj, simin, N_min, org_image_map = s.getSets()

    print "end of finding sets"

    w = Weight(N_min, sbmaj, simin, org_image_map)
    w.generateWeights()
    w.generateSWSP()
    I_w, S_w, S_p = w.getSets()

    print "end of finding the weights"
    print "len of weights ", len(I_w)