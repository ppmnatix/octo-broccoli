from utils import UTILS
from transformations import TRANSFORMATIONS
import numpy as np
from depth import TUKEY
from phij import PHIJ

class LOCAL_REPRESENTATION:

    def LocalRespresentacion(A,R,MuD,SigD,MuA,SigA,numPivots,J=1, phijsPath=""):
        try:
            Phis=[]
            for j in range(0,len(A)):
                px=A[j]
                pxAngle=UTILS.AngleBetweenPoints([0,0],px)
                #Get the points inside the neighborhood with radio R
                resp=UTILS.getRangeNeighbors(A,px,R,ignoreFirst=True)
                #Normalize the points on the studied point
                normResp=np.asarray([[o[0],o[1]-px[0],o[2]-px[1],o[3]] for o in resp])
                SigD=np.std([np.linalg.norm(o) for o in normResp])


                if(len(resp)>3):
                    #RangePolygon.printGraph(self,(0,0),normResp,[-R,R],[-R,R],R,'Anorm.pdf')
                    #TukeyDepth with ties
                    halfDepths=TUKEY.MaxTukeyDepthAll(normResp[:,[1,2]],[0,0])
                    tkAngle=-1.
                    for t in range(0,len(halfDepths)):
                        #angle=RangePolygon.getAngleBetweenPoints(self,[0,0],normResp[halfDepths[t][1]])
                        #Se rotan todos los puntos con el angulo necesario para hacer el segmento anterior el origen
                        angleToRotate= (2*np.pi)-halfDepths[t][3]
                        rotatedPoints=TRANSFORMATIONS.Rotations(normResp[:,[1,2]],angleToRotate)
                        tkAngle=pxAngle-UTILS.AngleBetweenPoints([0,0],[resp[halfDepths[t][2]][1],resp[halfDepths[t][2]][2]])
                        #Se generan los separadores por angulo
                        angleBuckets=UTILS.getAngleBuckets(numPivots)
                        #Se llena cada region con puntos que pertencen
                        #fBuckets=RangePolygon.getFilledBuckets(self,angleBuckets,rotatedPoints)
                        newCoords=[]
                        #Se agrega el origen como comienzo del poligono
                        #newCoords.append([0.,0.])
                        for i in range(0,len(angleBuckets)):
                            #newCoords.append([0,0])
                            newCoords.append(UTILS.getBucketWeights(rotatedPoints,angleBuckets[i],R,MuD,SigD,MuA,SigA))
                        srtcoord=sorted(newCoords, key=lambda neighbor: np.linalg.norm(neighbor))
                        Phij=PHIJ.getSimplePhij(srtcoord,J)
                        if(Phij!= None):
                            Phis.append([Phij.real,Phij.imag,tkAngle,halfDepths[t][2]])
                            #Phis.append([Phij.real,Phij.imag,math.fabs(RangePolygon.getAngleBetweenPoints(self,[0,0],resp[0][1])-RangePolygon.getAngleBetweenPoints(self,[0,0],resp[halfDepths[t][1]][1]))])

            return Phis
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")

    def LocalRespresentacionForPointClouds(tDb, R, MuD, SigD, MuA, SigA, numPivots,J, phijsPath=""):
        Phijs = []
        for i in range(0, len(tDb)):
            # if(i>1 and math.fmod(i,1000)==0):
            #   print(i)
            Phijs.append((i, LOCAL_REPRESENTATION.LocalRespresentacion(tDb[i][1], R, MuD, SigD, MuA, SigA, numPivots, 1,phijsPath)))
            if (phijsPath != ""):
                np.save(phijsPath + "/" + str(i) + ".npy", np.asarray(Phijs[i][1]))
        return Phijs


'''points = np.random.rand(100, 2) * 1000'''
'''
points= np.array([[224.17221103,292.42005961],
[939.45256212,999.32877867],
[397.62607376,649.00039169],
[912.71844447,207.25103598],
[169.35985114,148.77743817],
[575.50632674,237.95708782],
[57.09127784,73.30797895],
[831.39861999,151.0107567],
[9.75948296,453.42939095],
[306.86771745,497.96652585],
[16.17923466,643.55100189],
[743.22076779,262.23363918],
[893.97459271,669.42012926],
[218.54608067,165.69346429],
[926.32474988,62.36981456],
[781.0826384,532.93737672],
[230.36094575,409.31577482],
[448.64153142,623.00770405],
[205.63489351,37.37378191],
[818.78969695,389.29292831],
[700.20272773,321.53358891],
[927.02576146,620.05271118],
[598.05643175,649.84169606],
[47.55233228,815.8351307],
[641.05482157,935.54751086],
[950.99650616,442.43364373],
[570.70582363,798.56624952],
[405.62038973,495.06859675],
[86.94599492,989.77373958],
[703.04436497,363.46010151],
[914.21551269,248.65280413],
[189.93258028,850.48940402],
[403.4497003,380.16601801],
[45.70989174,991.38677149],
[77.02525201,678.72744647],
[598.43635499,354.92812072],
[709.54150613,681.18894408],
[497.13337992,155.86696976],
[532.17079064,890.54803909],
[658.21175968,892.9383408],
[378.56826706,770.06414947],
[920.47712519,519.96719795],
[455.66143495,281.87416437],
[991.82312883,792.9128052],
[554.5981105,57.01121211],
[614.35662925,857.80910706],
[108.40244099,263.06318527],
[621.67913706,510.23570158],
[89.39327734,976.13752821],
[963.94566816,523.27427644],
[188.35716333,87.8886185],
[227.83063313,235.41970113],
[918.7815147,630.9252246],
[65.88936667,939.74190497],
[979.8457392,524.98720743],
[91.93274589,659.72734142],
[584.81213536,224.8618186],
[168.8773948,737.9367324],
[531.31491812,585.47569076],
[774.10253511,259.44229952],
[519.71296395,403.87508342],
[203.91643968,194.21068958],
[914.76843915,98.15471338],
[211.77980231,740.02748248],
[606.84624161,914.88313725],
[300.21647535,387.52365329],
[316.1162231,512.67829719],
[841.71830799,374.41620759],
[409.52325017,810.97271942],
[636.1854566,933.25033152],
[454.39920675,31.88542339],
[534.15441075,413.46870195],
[332.33943403,777.02573264],
[742.23137508,163.05959936],
[827.48476174,444.47001678],
[436.08684653,381.6124532],
[319.84646939,942.28039825],
[711.42192042,212.39068433],
[36.3808178,820.72894837],
[136.82049158,963.72343008],
[943.66845731,637.09890087],
[183.0507477,911.37836455],
[11.76186173,107.51724024],
[497.864042,666.50796221],
[47.36162851,454.96979884],
[165.49214824,29.69504274],
[157.28992424,62.21839335],
[481.8540553,639.08068176],
[604.79045779,52.05036283],
[797.30276469,301.72992011],
[438.43275375,376.55094541],
[224.50733887,583.12322398],
[861.84309262,905.02769127],
[672.85185117,308.11866884],
[557.87287585,670.7389081],
[495.02290917,244.01068301],
[586.50219294,432.8234754],
[113.5061276,741.41022277],
[158.43913643,840.9421099],
[510.86872618,936.75176689]])

phisOrigs=LOCAL_REPRESENTATION.LocalRespresentacion(points,200,0,200,0,np.pi/2,32)

theta = 5.4
translationvector = [250, 30]
scalefactor = 2
pointsTransformed = TRANSFORMATIONS.Rotations(points, theta)
pointsTransformed=TRANSFORMATIONS.Tranlations(pointsTransformed,translationvector)
#pointsTransformed=TRANSFORMATIONS.Scales(pointsTransformed,scalefactor)

phisTrans=LOCAL_REPRESENTATION.LocalRespresentacion(pointsTransformed,200,0,200,0,np.pi/2,32)


print('exit 0')
'''