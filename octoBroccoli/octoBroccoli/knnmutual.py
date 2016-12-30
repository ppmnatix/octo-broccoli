import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import operator
import random
import matplotlib
from scipy.spatial import cKDTree

class MUTUALKNN:
    Points=None
    K=3
    minPointsClusters=5
    kdtree=None

    def __init__(self,mPoints,mK,mMinPointsClusters):
        self.Points=mPoints
        self.K=mK
        self.minPointsClusters=mMinPointsClusters
        self.kdtree=cKDTree(mPoints)

    def getKNNMatrix(self,trainingSet, k):
        NMatx=np.zeros((len(trainingSet),len(trainingSet)))
        LKNNS=[]
        for i in range(len(trainingSet)):
            #EL sumerle k+1 es para no mover mas coigo y solo se ignora el primero
            tmpKNNfoiInstance = self.kdtree.query(trainingSet[i], k=k+1)
            LKNNS.append(tmpKNNfoiInstance[1][1:len(tmpKNNfoiInstance[1])])
            for j in range(1,len(tmpKNNfoiInstance[1])):
                NMatx[i][tmpKNNfoiInstance[1][j]]=1
        return NMatx,LKNNS

    def getKNNInvMatrix(self,NMatx):
        NINVMatx=np.zeros_like(NMatx)
        for i in range(len(NMatx)):
            for j in range(len(NMatx[i])):
                if(NMatx[i][j]==1 and NMatx[j][i]==1):
                    NINVMatx[i][j]=1
                    NINVMatx[j][i]=1
        return NINVMatx

    def getConnectedComponents(self,NInvMtx):
        connectedComponents2 = nx.connected_components(nx.from_numpy_matrix(NInvMtx))
        lst = [c for c in sorted(connectedComponents2, key=len, reverse=True) if len(c)>self.minPointsClusters]
        return lst

    def getConnectedComponentsWithMutualKNN(self):
        NMtx, LKNNS = self.getKNNMatrix(self.Points, self.K)
        NInvMtx = self.getKNNInvMatrix(NMtx)
        Componenets = self.getConnectedComponents(NInvMtx)
        return Componenets, NInvMtx

'''
def main():

    A = np.random.randint(1000, size=(1000, 2))
    Experiment= MUTUALKNN(A,3,5)
    NMtx,LKNNS=Experiment.getKNNMatrix(A,3)
    NInvMtx=Experiment.getKNNInvMatrix(NMtx)
    Componenets=Experiment.getConnectedComponents(NInvMtx)
    print (Componenets)

main()
'''