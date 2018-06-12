
from PIL import Image
import random
import numpy
import pdb

from PIL import Image

import array
import logging

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math, bisect



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
    def __init__(self, oversampled_array, k=3, max_PCA_iterations=10, min_distance=5.0, size=200, m=2.5, epsilon=.5, max_FCM_iterations=100):
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

        self.oversampled_array = oversampled_array

    # Takes in an image and performs FCM Clustering.
    def run(self, image):
        self.image = image
        self.image.thumbnail(self.size)
        self.pixels2 = numpy.array(image.getdata(), dtype=numpy.uint8)
        # self.beta = self.calculate_beta(self.image)


        for i in range(self.s):
            self.pixels.append(self.pixels2[i])

        for i in range(1):
            for i in range(len(self.oversampled_array)):
                self.pixels.append(numpy.asarray(self.oversampled_array[i]))


        # set the size
        self.s_new = self.s + ( 1 *len(self.oversampled_array))
          
        print len(self.pixels), len(self.pixels2)
        print "********** smote array size  ", len(self.oversampled_array)

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

        self.showClustering("PCA.png") 

        for cluster in self.clusters:
            print cluster.centroid
        return [cluster.centroid for cluster in self.clusters]


    def selectSingleSolution(self):
        self.max_PCA_iterations = 10
        self.max_FCM_iterations=5


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
                    if self.calcDistance(self.gt_image_map[col_x,row_x], self.gt_image_map[157,16]) != 0.0: 
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
                    if self.calcDistance(self.gt_image_map[col_x,row_x], self.gt_image_map[157,16]) != 0.0: 
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
                        if self.calcDistance(self.gt_image_map[col_x,row_x], self.gt_image_map[157,16]) != 0.0: 
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
                        if self.calcDistance(self.gt_image_map[col_x,row_x], self.gt_image_map[157,16]) == 0.0: 
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
                            if self.calcDistance(self.gt_image_map[col_x,row_x], self.gt_image_map[157,16]) == 0.0: 
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
                if self.calcDistance(self.gt_image_map[i,j], self.gt_image_map[157,16]) == 0.0: 
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


class clustering(object):
    def __init__(self, N_min, smin, sminf, org_image_map):
        self.N_min = N_min
        self.smin = smin
        self.sminf = sminf
        self.org_image_map = org_image_map
        self.L = []
        self.which_cluster = []
        self.C_p = 3.0
        self.d_avg = 0
        self.threshold = 0
       
    def findThresholdAvg(self):
        d_avg = 0

        for i in self.sminf:
            tmp = []

            for j in self.sminf:
                if i == j:
                    continue

                distance = self.calcEucledianDistance(i, j)
                tmp.append(distance)

                print distance
            self.d_avg +=min(tmp)
        self.d_avg/=len(self.sminf)
        self.threshold = self.d_avg * self.C_p

        print self.d_avg, self.threshold

    def run_cluster(self):
        self.L = { index:[i] for index, i in enumerate(self.smin)}

        print len(self.smin), len(self.L)

        clusters_number = range(len(self.smin))

        dis_table = [ [0 for i in clusters_number] for j in clusters_number]

        for i in clusters_number:
            for j in clusters_number:

                # print L[i][0], L[j]
                dis_table[i][j] = self.calcEucledianDistance(self.L[i][0], self.L[j][0])
                # print "distance ", dis_table[i][j]

        MAX = max(max(j) for j in dis_table)

        print "max ", MAX

        for i in clusters_number:
            dis_table[i][i] = MAX

        for i in self.smin:
            MIN = min(min(j) for j in dis_table)

            if MIN > self.threshold:
                break
            for j in clusters_number:
              if MIN in dis_table[j]:
                b = dis_table[j].index(MIN)
                a = j
                break
            self.L[a].extend(self.L[b])

            del self.L[b]
            clusters_number.remove(b)
            for j in clusters_number:
                tmp = self.calcEucledianDistance(self.L[a][0], self.L[j][0])
                dis_table[a][j] = tmp
                dis_table[j][a] = tmp

            dis_table[a][a] = MAX

            for j in clusters_number:
                dis_table[b][j] = MAX
                dis_table[j][b] = MAX
  
        self.which_cluster = {}
        for i, clu in self.L.items():
            for j in clu:
                self.which_cluster[j] = i


        print len(self.L), len(self.which_cluster)


        



    def getSets(self):
        return self.L, self.which_cluster


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


class WeightedSampleRandomGenerator(object):
    def __init__(self, indices, weights):
        self.totals = []
        self.indices = indices
        running_total = 0

        for w in weights:
            running_total += w
            self.totals.append(running_total)

    def next(self):
        rnd = random.random() * self.totals[-1]

        return self.indices[bisect.bisect_right(self.totals, rnd)]

    def __call__(self):
        return self.next()


class Resampling(object):
    def __init__(self, S_w, S_p, L, which_cluster, N, org_image_map):
        self.S_w = S_w
        self.S_p = S_p
        self.L = L
        self.which_cluster = which_cluster
        self.N = N
        self.org_image_map = org_image_map
        self.X_gen = []

    def resample(self):
        self.X_gen = []
        some_big_number = 10000000.
        sample = WeightedSampleRandomGenerator(self.S_w.keys(), self.S_w.values())

        for i in range(self.N):
            x = sample()
            y = random.choice( self.L[self.which_cluster[x]] )
            # print y

            alpha = random.randint(0, some_big_number) / some_big_number

            dis = self.calcEucledianDistance(x, y)

            factor = dis * alpha
            s = self.get_pixel_value(x) 

            print s, factor
            s = [s[0]+factor, s[1] + factor, s[2]+factor , 255] 
            print s
            self.X_gen.append(s)

        return self.X_gen



    def get_pixel_value(self, loc):
        return self.org_image_map[loc[0], loc[1]]





            # print x
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


if __name__ == "__main__":
    s= Set("T4.png", "GT_T4.png")
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


    #step 10

    c = clustering(N_min, simin, sminf, org_image_map)
    c.findThresholdAvg()
    c.run_cluster()
    L, which_cluster = c.getSets()


    r = Resampling(S_w, S_p, L, which_cluster, 100000, org_image_map)
    X_gen = r.resample()

    print "generated", len(X_gen)

    f = fcm(X_gen)
    image = Image.open("T4.png")
    f.run(image)