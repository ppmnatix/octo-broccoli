from scipy.spatial import cKDTree
import pickle
import numpy as np
class INDEXING:

    def buildKDTree(PhijsO, filename):
        phiJs=np.ndarray([0, 2])
        #phiJsT= np.ndarray([0, 2])
        for tt in range(len(PhijsO)):
            phiJs=np.concatenate((phiJs, PhijsO[tt][1]), axis=0)

        kdt=cKDTree(phiJs)
        raw=pickle.dump(kdt,open(filename, 'wb'))

    def loadKDTree(filename):
        f = open(filename, 'rb')
        return pickle.load(f)

    def buildInvertedPhijs(PhijsO,filename):
        phiJs = []
        for tt in range(len(PhijsO)):
            phiJs = phiJs + [(PhijsO[tt][0], t[0],t[1]) for t in PhijsO[tt][1]]
        np.save(filename, np.asarray(phiJs))


    def loadInvertedPhijs( filename):
        return np.load(filename)