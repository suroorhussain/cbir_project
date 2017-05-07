import numpy as np
import csv
import os

from scipy.spatial import distance as dist

import colordescriptor
import zernikemoments

class Searcher:
    def __init__(self, indexPath):
        self.indexPath = indexPath
        
    def chi2_distance(self, histA, histB, eps = 1e-10):
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                    for (a, b) in zip(histA, histB)])
        return d

    def cdsearch(self, image, limit = 5):
        #describe the image
        cd = colordescriptor.ColorDescriptor((8, 12, 3))
        queryfeatures = cd.describe(image)
        #initialize the dictionary of results
        results = {}
        #open the index file for reading
        with open(os.path.join(self.indexPath, 'colordescriptor.csv')) as f:
            #initialize the CSV reader
            reader = csv.reader(f)

            for row in reader:
                #parse out the image ID and features, then compute the
                #chi-squared distance between the features in our index
                #and our query features
                features = [float(x) for x in row[1:]]
                d = self.chi2_distance(features, queryfeatures)

                results[row[0]] = d

            #sort out the results so that the smaller distances are at
            #the fron of the list
            results = sorted([(v,k) for (k, v) in results.items()])
            
            return results[:limit]

    def zmsearch(self, image, limit=5):
        #describe the query image
        zm = zernikemoments.ZernikeMoments(25)
        queryfeatures = zm.describe(image)

        #initialize the dictionary of results
        results = {}
        #open the index file for reading
        with open(os.path.join(self.indexPath, 'zernikemoment.csv')) as f:
            #initialize the CSV reader
            reader = csv.reader(f)

            # loop over the images in our index
            for row in reader:
                # compute the distance between the query features
                # and features in our index, then update the results
                features = [float(x) for x in row[1:]]
                d = dist.euclidean(queryfeatures, features)
                results[row[0]] = d

            # sort our results, where a smaller distance indicates
            # higher similarity
            results = sorted([(v, k) for (k, v) in results.items()])[:limit]

            # return the results
            return results

    def search(self, image):
        cdresult = self.cdsearch(image)
        cdresult = [imagepath for (score, imagepath) in cdresult]
        cdresult = set(cdresult)
        zmresult = self.zmsearch(image)
        zmresult = [imagepath for (score, imagepath) in zmresult]
        zmresult = set(zmresult)
        result = cdresult.union(zmresult)

        return result
