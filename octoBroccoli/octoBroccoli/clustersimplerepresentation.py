from utils import UTILS
from transformations import TRANSFORMATIONS
import numpy as np
from depth import TUKEY
from phij import PHIJ
from knnmutual import MUTUALKNN
from Distortions import Distortions

class CLUSTER_SIMPLE_REPRESENTATION:
    def ClusterSimpleRepresentation(A, K, minPointsCluster, phijsPath=""):
        scaleConstant = 1000
        Resp = np.ndarray([0, 2])
        TukeysList = []
        # Generate the connected components for points
        objMutualKNN = MUTUALKNN(A, K, minPointsCluster)
        connectedComponents, NInvMtx = objMutualKNN.getConnectedComponentsWithMutualKNN()

        # for j in range(0,1):
        for j in range(0, len(connectedComponents)):

            componentX = [A[c] for c in (connectedComponents[j])]

            if (len(componentX) > minPointsCluster):

                # TukeyDepth with ties

                halfDepths = TUKEY.MaxTukeyDepthAll(componentX, [0, 0])
                for t in range(0, len(halfDepths)):
                    TukeysList.append([componentX[int(halfDepths[t][1])], componentX[int(halfDepths[t][2])]])

                    newPoints = np.asarray(componentX)

                    # Trasladar utilizand el punto Tukey como origen
                    translationvector = [componentX[int(halfDepths[t][2])][0], componentX[int(halfDepths[t][2])][1]]

                    newPoints[:, 0] -= translationvector[0]
                    newPoints[:, 1] -= translationvector[1]

                    # Rotate the points with tukey angle
                    tkLineAngle = -1.
                    tkLineAngle = UTILS.AngleBetweenPoints(newPoints[int(halfDepths[t][2])],newPoints[int(halfDepths[t][1])])
                    angleToRotate = (2 * np.pi) - tkLineAngle

                    # Se rotan todos los puntos con el angulo necesario para hacer el segmento anterior el origen
                    newPoints = TRANSFORMATIONS.Rotations(newPoints, angleToRotate)

                    # Escalar puntos
                    scaleFactor = scaleConstant / (np.linalg.norm([newPoints[int(halfDepths[t][1])] - newPoints[int(halfDepths[t][2])]]))
                    newPoints[:, 0] *= scaleFactor
                    newPoints[:, 1] *= scaleFactor

                    # Se quitan los puntos que forman la recta
                    newPoints = np.delete(newPoints, [int(halfDepths[t][1]), int(halfDepths[t][2])], 0)

                    Resp = np.concatenate((Resp, newPoints), axis=0)

        return Resp, NInvMtx, TukeysList


    def ClusterSimpleRepresentationForPointClouds(tDb, K, minPointsCluster,phijsPath=""):
        Phijs = []
        for i in range(0, len(tDb)):
            Phijs, matrixMKNN,Tukeys= CLUSTER_SIMPLE_REPRESENTATION.ClusterSimpleRepresentation(tDb[i][1], K, minPointsCluster, )
            if (phijsPath != ""):
                np.save(phijsPath + "/" + str(i) + ".npy", np.asarray(Phijs))
        return Phijs