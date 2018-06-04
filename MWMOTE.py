
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math


class Cluster(object):
    # Constructor for cluster object
    def __init__(self):
        self.pixels = []  # intialize pixels into a list
        self.centroid = None  # set the number of centro
        # ids to none

    def addPoint(self, pixel):  # add pixels to the pixel list
        self.pixels.append(pixel)


class fcm(object):
    # __inti__ is the constructor and self refers to the current object.
    def __init__(self, k=3, max_PCA_iterations=10, min_distance=5.0, size=100, m=2.5, epsilon=.5, max_FCM_iterations=100):
        self.k = k  # initialize k clusters

        # intialize max_iterations
        self.max_PCA_iterations = max_PCA_iterations
        self.max_FCM_iterations = max_FCM_iterations

        self.min_distance = min_distance  # intialize min_distance
        self.degree_of_membership = []
        self.s = size ** 2
        self.size = (size, size)  # intialize the size
        self.m = m
        self.epsilon = 0.01
        self.max_diff = 10.0
        self.image = 0
        self.pixels2 = []
        self.pixels = []
        self.s_new = self.s 

        self.SMOTE_array = []

    # Takes in an image and performs FCM Clustering.
    def run(self, image):
        self.image = image
        self.image.thumbnail(self.size)
        self.pixels2 = numpy.array(image.getdata(), dtype=numpy.uint8)
        # self.beta = self.calculate_beta(self.image)

        
        brown = (21 * 100) + 33


        for i in range(self.s):
            self.pixels.append(self.pixels2[i])

        
        for i in range(len(self.SMOTE_array)):
            self.pixels.append(numpy.asarray(self.SMOTE_array[i]))


        # set the size
        self.s_new = self.s + ( 1 *len(self.SMOTE_array))
          
        print len(self.pixels), len(self.pixels2)
        print "********** smote array size  ", len(self.SMOTE_array)

        self.clusters = [None for i in range(self.k)]
        self.oldClusters = None

        for i in range(self.s_new):
            self.degree_of_membership.append(numpy.random.dirichlet(numpy.ones(self.k), size=1))

        for i in range(self.s_new):
            num_1 = random.randint(1, 2) * 0.1
            num_2 = random.randint(1, 2) * 0.1
           
            num_3 = 1.0 - (num_1+num_2)
            degreelist = [num_1, num_2, num_3]
            self.degree_of_membership[i] = degreelist

        randomPixels = random.sample(self.pixels, self.k)
        print"INTIALIZE RANDOM PIXELS AS CENTROIDS"
        print randomPixels
        #    print"================================================================================"
        for idx in range(self.k):
            self.clusters[idx] = Cluster()
            self.clusters[idx].centroid = randomPixels[idx]
            # if(i ==0):
        for cluster in self.clusters:
            for pixel in self.pixels:
                cluster.addPoint(pixel)

        print "________", self.clusters[0].pixels[0]
        iterations = 0

        # FCM
        while self.shouldExitFCM(iterations) is False:
            self.oldClusters = [cluster.centroid for cluster in self.clusters]
            print "HELLO I A AM ITERATIONS:", iterations
            print"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            self.calculate_centre_vector()

            self.update_degree_of_membershipFCM()
           

            iterations += 1

        iterations = 0
        self.showClustering("FCM.png")
        # self.DB_index()

        # PCA
        while self.shouldExitPCA(iterations) is False:
            self.oldClusters = [cluster.centroid for cluster in self.clusters]
            print "HELLO I A AM ITERATIONS:", iterations
            print"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            self.calculate_centre_vector()

            self.update_degree_of_membershipPCA()
            
            iterations += 1

        for cluster in self.clusters:
            print cluster.centroid
        return [cluster.centroid for cluster in self.clusters]


    def selectSingleSolution(self):
        self.max_PCA_iterations = 10
        self.max_FCM_iterations=5


    def getClusterCentroid(self):
        centroid = []
        for cluster in self.clusters:
            centroid.append(cluster.centroid);

        return centroid

    def printClustorCentroid(self):
        for cluster in self.clusters:
            print cluster.centroid

    def shouldExitFCM(self, iterations):
        if self.max_diff<self.epsilon:
            return True

        if iterations <= self.max_FCM_iterations:
            return False
        return True

    def shouldExitPCA(self, iterations):
        if iterations <= self.max_PCA_iterations:
            return False
        return True

    # Euclidean distance (Distance Metric).
    def calcDistance(self, a, b):
        result = numpy.sqrt(sum((a - b) ** 2))
        return result

    # Calculates the centroids using degree of membership and fuzziness.
    def calculate_centre_vector(self):
        for cluster in range(self.k):
            sum_numerator = 0.0
            sum_denominator = 0.0
            for i in range(self.s_new):
                pow_uij= pow(self.degree_of_membership[i][cluster], self.m)
                sum_denominator +=pow_uij
                num= pow_uij * self.pixels[i]

                sum_numerator+=num

            updatedcluster_center = sum_numerator/sum_denominator

            self.clusters[cluster].centroid = updatedcluster_center


    # Updates the degree of membership for all of the data points.
    def update_degree_of_membershipFCM(self):
        self.max_diff = 0.0

        for idx in range(self.k):
            for i in range(self.s_new):
                new_uij = self.get_new_value(self.pixels[i], self.clusters[idx].centroid)
                if (i == 0):
                    print "This is the Updatedegree centroid number:", idx, self.clusters[idx].centroid
                diff = new_uij - self.degree_of_membership[i][idx]
                if (diff > self.max_diff):
                    self.max_diff = diff
                self.degree_of_membership[i][idx] = new_uij
        return self.max_diff

    def get_new_value(self, i, j):
        sum = 0.0
        val = 0.0
        p = (2 * (1.0) / (self.m - 1))  # cast to float value or else will round to nearst int
        for k in self.clusters:
            num = self.calcDistance(i, j)
            denom = self.calcDistance(i, k.centroid)
            val = num / denom
            val = pow(val, p)
            sum += val
        return (1.0 / sum)

    def getEta(self, idx):
        sum_membership = 0.0
        eta_numerator = 0.0
        eta_k = 1.0

        for i in range(self.s_new):
            dis = pow(self.calcDistance(self.clusters[idx].centroid, self.pixels[i]), 2.0)

            membership_power = pow(self.degree_of_membership[i][idx], self.m)

            eta_numerator += (membership_power * dis)

            sum_membership += membership_power

        eta =eta_numerator / sum_membership
        eta = eta * eta_k
        return eta

    # update the degree of membership for PCA
    def update_degree_of_membershipPCA(self):
        #PCA 96
        for idx in range(self.k):
            eta = 0.0
            eta_k = 1.0

            #get eta for particular cluster
            eta = self.getEta(idx)

            if eta > 0.0:
                # print "******************* eta", eta
                for i in range(self.s_new):
                    if (i == 0):
                        print "This is the Update degree centroid number:", idx, self.clusters[idx].centroid

                    dis = pow(self.calcDistance(self.clusters[idx].centroid, self.pixels[i]), 2.0)

                    factor = dis / eta
                    factor = factor * -1.0

                    updated_membership_degree = math.exp(factor)

                    self.degree_of_membership[i][idx] = updated_membership_degree

    def SMOTE(self):

        #open first image
        image_GT = Image.open('GT_T3.png')
        pixels_GT = numpy.array(image_GT.getdata(), dtype=numpy.uint8)


        #open second image
        image_org = Image.open('T3.png')
        pixels_org = numpy.array(image_org.getdata(), dtype=numpy.uint8)


        GT_pixelmap = image_GT.load()
        Org_pixelmap  = image_org.load()

        factor = 0.1
      #get white pixel
        j1 = (100 * 22) + 41


        count =0

        for i in range(100):
            for j in range(100):
                if self.calcDistance(GT_pixelmap[i,j], pixels_GT[j1]) == 0.0: 

                    pixel = [int(Org_pixelmap[i, j][0]) , int(Org_pixelmap[i, j][1] ), int(Org_pixelmap[i, j][2]), 255]
                    self.SMOTE_array.append(pixel)

                    ## only selecting the village area
                    count+=1
                    #SMOTE

                    #get their four neighbours
                        #  y-1, x 
                        #if this pixel is also belongs to village

                    if i>0 and j>0 and i<99 and j<99:
                        if self.calcDistance(GT_pixelmap[i-1, j], pixels_GT[j1]) == 0.0:
                        
                        #take the difference
                            print ""#, self.pixelmap[i-1, j], self.pixelmap[i, j]
                            r = (Org_pixelmap[i-1, j][0]-Org_pixelmap[i, j][0])
                            g = (Org_pixelmap[i-1, j][1]-Org_pixelmap[i, j][1])
                            b = (Org_pixelmap[i-1, j][2]-Org_pixelmap[i, j][2])

                            print r,g,b
                            print r*0.5 , g * 0.5 , b * 0.5
                            print Org_pixelmap[i,j]
                            pixel = [int(Org_pixelmap[i, j][0] + (r *factor)) , int(Org_pixelmap[i, j][1] + (g *factor)), int(Org_pixelmap[i, j][2] +(b*factor)), 255]
                            self.SMOTE_array.append(pixel)
                            print pixel 

                        if self.calcDistance(GT_pixelmap[i+1, j], pixels_GT[j1]) == 0.0:
                        
                        #take the difference
                            print ""#, self.pixelmap[i-1, j], self.pixelmap[i, j]
                            r = (Org_pixelmap[i+1, j][0]-Org_pixelmap[i, j][0])
                            g = (Org_pixelmap[i+1, j][1]-Org_pixelmap[i, j][1])
                            b = (Org_pixelmap[i+1, j][2]-Org_pixelmap[i, j][2])

                            print r,g,b
                            print r*0.5 , g * 0.5 , b * 0.5
                            print Org_pixelmap[i,j]
                            pixel = [int(Org_pixelmap[i, j][0] + (r *factor)) , int(Org_pixelmap[i, j][1] + (g *factor)), int(Org_pixelmap[i, j][2] +(b*factor)), 255]
                            self.SMOTE_array.append(pixel)
                            print pixel 


                        if self.calcDistance(GT_pixelmap[i, j-1], pixels_GT[j1]) == 0.0:
                        
                        #take the difference
                            print ""#, self.pixelmap[i-1, j], self.pixelmap[i, j]
                            r = (Org_pixelmap[i, j-1][0]-Org_pixelmap[i, j][0])
                            g = (Org_pixelmap[i, j-1][1]-Org_pixelmap[i, j][1])
                            b = (Org_pixelmap[i, j-1][2]-Org_pixelmap[i, j][2])

                            print r,g,b
                            print r*0.5 , g * 0.5 , b * 0.5
                            print Org_pixelmap[i,j]
                            pixel = [int(Org_pixelmap[i, j][0] + (r *factor)) , int(Org_pixelmap[i, j][1] + (g *factor)), int(Org_pixelmap[i, j][2] +(b*factor)), 255]
                            self.SMOTE_array.append(pixel)
                            print pixel 


                        if self.calcDistance(GT_pixelmap[i, j+1], pixels_GT[j1]) == 0.0:
                        
                        #take the difference
                            print ""#, self.pixelmap[i-1, j], self.pixelmap[i, j]
                            r = (Org_pixelmap[i, j+1][0]-Org_pixelmap[i, j][0])
                            g = (Org_pixelmap[i, j+1][1]-Org_pixelmap[i, j][1])
                            b = (Org_pixelmap[i, j+1][2]-Org_pixelmap[i, j][2])

                            print r,g,b
                            print r*0.5 , g * 0.5 , b * 0.5
                            print Org_pixelmap[i,j]
                            pixel = [int(Org_pixelmap[i, j][0] + (r *factor)) , int(Org_pixelmap[i, j][1] + (g *factor)), int(Org_pixelmap[i, j][2] +(b*factor)), 255]
                            self.SMOTE_array.append(pixel)
                            print pixel 


                        # and take its 0.5
                        # self.pixelmap[i,j] = (255, 0, 0)

        print len(self.SMOTE_array), count
        print "******************** ", len(self.SMOTE_array)



    def showClustering(self, name):
        localPixels = [None] * len(self.image.getdata())
        for idx, pixel in enumerate(self.pixels2):
            shortest = float('Inf')
            for cluster in self.clusters:
                distance = self.calcDistance(cluster.centroid, pixel)
                if distance < shortest:
                    shortest = distance
                    nearest = cluster

            # if nearest == self.clusters[0]:
            #     localPixels[idx]=[229,75,77]
            # elif nearest == self.clusters[1]:
            #     localPixels[idx] = [56,129,78]
            # elif nearest == self.clusters[2]:
            #     localPixels[idx] = [251,227,227]
            localPixels[idx] = nearest.centroid

        w, h = self.image.size
        localPixels = numpy.asarray(localPixels) \
            .astype('uint8') \
            .reshape((h, w, 4))
        colourMap = Image.fromarray(localPixels)
        # colourMap.show()

        plt.imsave(name, colourMap)


    def normalization(self):
        for i in range(self.s):
            max = 0.0
            highest_index = 0
            # Find the index with highest probability
            for j in range(self.k):
                if (self.degree_of_membership[i][j] > max):
                    max = self.degree_of_membership[i][j]
                    highest_index = j
            # Normalize, set highest prob to 1 rest to zero
            for j in range(self.k):

                if (j != highest_index):
                    self.degree_of_membership[i][j] = 0
                else:
                    self.degree_of_membership[i][j] = 1

    def getVariance(self, cluster):
        mean=0
        sum = 0.0
        no_of_pixels = 0
        r = 2.0
        for i in range(self.s):
            if self.degree_of_membership[i][cluster] == 1:
                sum+=self.pixels[i]
                no_of_pixels +=1

        # calculate the mean
        mean = sum/no_of_pixels

        #calculate the variance

        sum_of_distance = 0.0
        for i in range(self.s):
            if self.degree_of_membership[i][cluster] == 1:
                dis = self.calcDistance(self.pixels[i], mean)
                sum_of_distance += pow(dis,r)

        x = sum_of_distance / no_of_pixels
        var = pow(x,1.0/r)

        print " -------- ", var, x, sum_of_distance

        return var


    def DB_index(self):

        # get the maximum of the Rij
        max_sum = 0.0

        for i in range(self.k):

            r_list = []
            for j in range(self.k):
                if i!=j:
                    var1 = self.getVariance(i)
                    var2 = self.getVariance(j)

                    sum_var = var1+var2
                    dis_cluster_center = self.calcDistance(self.clusters[i].centroid, self.clusters[j].centroid)
                    r_ij = sum_var/dis_cluster_center
                    r_list.append(r_ij)

            print ">>>>>>>> ", r_list

            #get the max Rij from list
            list.sort(r_list)
            print ">>>>>>>> sorted ", r_list, r_list[-1]

            #get the max of Rij, and store it
            max_sum +=r_list[-1]

        db = max_sum/self.k
        print "DB index ", db

        return db



if __name__ == "__main__":
    image = Image.open("T3.png")
    f = fcm()

    f.SMOTE()
    result = f.run(image)


    f.showClustering("PCA.png")

    f.normalization()
    f.DB_index()
    # # print f.I_index()
    # # # print f.JmFunction()
    # # print f.XBindex()

    # f.normalization()
    # print f.DB_index()
