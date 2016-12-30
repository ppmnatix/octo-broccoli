from utils import UTILS
from transformations import TRANSFORMATIONS
import numpy as np
from depth import TUKEY
from phij import PHIJ
from knnmutual import MUTUALKNN
from Distortions import Distortions

class CLUSTER_REPRESENTATION:
    def ClusterRepresentation(A,K,minPointsCluster,MuD,SigD,MuA,SigA,numPivots,J=1, phijsPath=""):
        try:
            Phis=[]
            TukeysList=[]
            #Generate the connected components for points
            objMutualKNN = MUTUALKNN(A, K, minPointsCluster)
            connectedComponents, NInvMtx=objMutualKNN.getConnectedComponentsWithMutualKNN()

            # Se generan los separadores por angulo
            angleBuckets = UTILS.getAngleBuckets(numPivots)

            for j in range(0,len(connectedComponents)):

                componentX= [A[c] for c in (connectedComponents[j])]

                if(len(componentX)>3):

                    #TukeyDepth with ties
                    halfDepths=TUKEY.MaxTukeyDepthAll(componentX,[0,0])
                    for t in range(0,len(halfDepths)):
                        tkLineAngle = -1.
                        tkLineAngle = UTILS.AngleBetweenPoints(componentX[int(halfDepths[t][1])], componentX[int(halfDepths[t][2])])
                        TukeysList.append([componentX[int(halfDepths[t][1])], componentX[int(halfDepths[t][2])]])
                        angleToRotate = (2 * np.pi) -tkLineAngle
                        #Se rotan todos los puntos con el angulo necesario para hacer el segmento anterior el origen
                        rotatedPoints = TRANSFORMATIONS.Rotations(componentX, angleToRotate)
                        # Normalize the points on the studied point
                        normResp = UTILS.normalizeData(rotatedPoints, 0, 100)
                        # normResp=np.asarray([[o[0]-rotatedPoints[int(halfDepths[t][2])][0],o[1]-rotatedPoints[int(halfDepths[t][2])][1]] for o in rotatedPoints])

                        normResp = np.asarray([[o[0] - 50, o[1] - 50] for o in normResp])

                        MuD=0
                        #SigD=50
                        SigD=np.std([np.linalg.norm(o) for o in normResp])
                        SigA = np.std([UTILS.AngleBetweenPointsPITO_PI([0,0],o) for o in normResp])


                        newCoords=[]
                        #Se agrega el origen como comienzo del poligono
                        #newCoords.append([0.,0.])
                        for i in range(0,len(angleBuckets)):
                            #newCoords.append([0,0])
                            newCoords.append(UTILS.getBucketWeights(normResp,angleBuckets[i],50,MuD,SigD,MuA,SigA))
                        srtcoord=sorted(newCoords, key=lambda neighbor: np.linalg.norm(neighbor))
                        Phij=PHIJ.getPhij(srtcoord,J)
                        if(Phij!= None):
                            Phis.append([Phij.real,Phij.imag])

            return Phis,NInvMtx,TukeysList
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")

    def ClusterRepresentationForPointClouds(tDb,K,minPointsCluster,MuD,SigD,MuA,SigA,numPivots,J=1, phijsPath=""):
        Phijs = []
        for i in range(0, len(tDb)):
            Phijs.append((i, CLUSTER_REPRESENTATION.ClusterRepresentation(tDb[i][1],K,minPointsCluster,MuD,SigD,MuA,SigA,numPivots,J=1, phijsPath="")))
            if (phijsPath != ""):
                np.save(phijsPath + "/" + str(i) + ".npy", np.asarray(Phijs[i][1]))
        return Phijs


'''
points= np.array([[ 193.62301224,233.09502592],
 [ 584.26466477,780.79523122],
 [ 956.75932252,459.70541583],
 [ 706.63906951,732.51113087],
 [ 808.97513328,839.65380132],
 [ 177.87063166,792.18666092],
 [ 275.77554452,551.20016173],
 [ 785.67491334,266.5307727 ],
 [68.64853491,865.558758,],
 [ 108.19420695,729.0161369 ],
 [ 837.45801386,660.92285869],
 [35.12651045,244.20240473],
 [ 994.87545023,598.80982368],
 [ 923.44055751, 13.71509589],
 [ 219.79938407,457.87415348],
 [ 452.79524028,593.50313614],
 [ 438.57021454,336.74555386],
 [ 707.9129131, 840.07903521],
 [ 3.05115648,537.7777936 ],
 [ 596.12723541,711.7798918 ],
 [78.77760965,405.75592451],
 [82.94148876,356.57249056],
 [ 217.82903289,704.17678657],
 [ 703.5239874, 659.13493054],
 [ 926.53662207,174.42867422],
 [ 621.49274766,896.04879212],
 [ 943.98571284,526.24193342],
 [69.12119865,488.73536017],
 [ 481.3036721, 502.60844184],
 [ 955.4157618, 961.70060754],
 [ 240.7443802, 999.59243024],
 [ 891.61517492,503.90157835],
 [40.07512576,397.68740842],
 [ 532.5054561, 548.47581824],
 [ 372.44135736,987.46081654],
 [ 922.80695588,626.36761533],
 [ 685.06597671,738.97521368],
 [ 924.20478572,798.57490947],
 [ 459.19130007,208.16426843],
 [ 916.7858993, 703.6653164 ],
 [ 441.4063728, 910.29963436],
 [ 545.45206574,345.14433144],
 [ 304.76851695,976.75855474],
 [ 147.76857187,116.64607771],
 [ 626.039989, 84.24685307],
 [73.1755866, 820.40765588],
 [55.17830961,846.03605628],
 [ 532.85803673,203.96126798],
 [ 349.39790839,726.43200415],
 [69.5089345, 851.65565293],
 [ 789.4075361, 287.52378541],
 [ 712.22667304,549.93603856],
 [ 374.37865444,491.50037127],
 [ 133.62137512,303.31073875],
 [ 754.66763032,200.54450314],
 [ 629.83201687,772.90719084],
 [ 422.99632349,499.55918497],
 [ 255.23446031,811.54376161],
 [ 274.02475613,177.43718839],
 [ 878.09458421, 33.56059619],
 [ 526.30650524,539.79969798],
 [ 432.8631739, 684.46297603],
 [ 416.35895449, 64.84903992],
 [ 672.20069504,776.97584146],
 [ 868.26304362,477.27229633],
 [ 144.65777846,236.15345775],
 [ 230.4882863, 755.26730476],
 [ 825.11277796, 62.4020099 ],
 [ 968.92653519, 35.96085649],
 [ 421.19776216, 40.53326373],
 [ 270.94557657,154.2464787 ],
 [ 565.50124327, 36.31150609],
 [ 528.43646794,246.69878608],
 [11.16090087,7.15546552],
 [56.90197894,762.13745795],
 [ 894.00862328,714.43352254],
 [ 266.548102,350.88328536],
 [ 454.45992774, 84.20085125],
 [ 400.32260089,751.79248926],
 [99.10801936,814.82327431],
 [ 271.29958975,166.74508581],
 [ 660.42889993,288.34473073],
 [ 664.87130255,365.27311972],
 [ 215.67590236,970.68728204],
 [ 687.16133523,786.5202399 ],
 [ 936.22262039,457.69720791],
 [ 893.88322895,136.00881569],
 [ 179.61599787,984.96933571],
 [ 920.36014803,253.83481084],
 [ 683.61670378,536.49416143],
 [ 644.66578341,749.25928129],
 [ 589.74977225,367.93540786],
 [ 278.83287055,581.15928227],
 [ 452.75909359,3.58449655],
 [ 633.80993375,898.08290182],
 [ 679.65314568,440.11112979],
 [ 615.7131561, 211.02283869],
 [ 313.79905834,585.00941613],
 [ 411.29074715,9.39180123],
 [ 265.9655757, 261.06062855]])



#points = np.random.rand(100, 2) * 1000
phisOrigs,NInvMtx,TukeysList=CLUSTER_REPRESENTATION.ClusterRepresentation(points,3,5,0,200,0,np.pi/2,32)


theta = 5.4
translationvector = [250, 30]
scalefactor = 2
pointsTransformed = TRANSFORMATIONS.Rotations(points, theta)
pointsTransformed=TRANSFORMATIONS.Tranlations(pointsTransformed,translationvector)
pointsTransformed=TRANSFORMATIONS.Scales(pointsTransformed,scalefactor)


noisyPoints=Distortions.Noise(pointsTransformed,10)
occlutedPoints=Distortions.Occlution(noisyPoints,10)

phisTrans,NInvMtx2,TukeysList2=CLUSTER_REPRESENTATION.ClusterRepresentation(occlutedPoints,3,5,0,200,0,np.pi/2,32)

testphis = []
for i in range(0, 10):
    phisOrigsym, NInvMtxt, TukeysListtm = CLUSTER_REPRESENTATION.ClusterRepresentation(np.random.rand(100, 2) * 1000, 3, 5, 0, 200, 0, np.pi / 2, 32)
    testphis=testphis+phisOrigsym



print(phisOrigs)
print('---------------')
print(phisTrans)
'''