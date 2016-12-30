import numpy as np
import math
from utils import UTILS
from transformations import TRANSFORMATIONS

class TUKEY:
    def TukeyRecursive(points,px):
        resp=[]
        tukeyDepths=TUKEY.MaxTukeyDepth(points,px)
        tukeyDepthsRecs=np.array([[0, row[1],-1] for row in tukeyDepths])
        maxSecTukey=-1
        for i in range(len(tukeyDepthsRecs)):
            tukeyDepthsRecs[i][0]=TUKEY.TukeyDepth(points, points[tukeyDepthsRecs[i][1]])
            tukeyDepthsRecs[i][2]=i

        tukeyDepthsRecs = tukeyDepthsRecs[np.argsort(tukeyDepthsRecs[:, 0],kind='mergesort')]
        idx=np.argmax(tukeyDepthsRecs ,axis=0 )[0]
        for i in range(idx,len(tukeyDepthsRecs)):
            resp.append(tukeyDepths[tukeyDepthsRecs[i][2]])
        return resp

    def MaxTukeyDepthAll(points,px):
        #resultstks=np.empty([len(points),3])
        resultstks=[]
        resp=[]
        for i in range(len(points)):
            tki=TUKEY.TukeyDepth(points,points[i])
            for j in range(0,len(tki)):
                resultstks.append([tki[j][0],tki[j][1],i])

                '''resultstks[i][0]=tki[j][0]
                resultstks[i][1] = tki[j][1]
                resultstks[i][2]=i'''
        resultstks=np.asanyarray(resultstks)
        resultstks = resultstks[np.argsort(resultstks[:, 0], kind='mergesort')]
        idx = np.argmax(resultstks, axis=0)[0]
        for i in range(idx, len(resultstks)):
            resp.append([resultstks[i][0], resultstks[i][1], resultstks[i][2]])
            #resp.append([resultstks[i][0],resultstks[i][1], resultstks[i][2], math.fabs(UTILS.AngleBetweenPoints(px,points[int(resultstks[i][2])]))])


        return resp

    def TukeyDepth(points, px):
        resp=[]
        eps=.000001
        n=len(points)
        tmpthetas=np.empty([n,2])
        resp=[]
        nt=0
        pxAngle=UTILS.AngleBetweenPoints([0,0],px)
        for i in range(0,n):
            anglebt=UTILS.AngleBetweenPoints(px,points[i])
            if(anglebt<=eps):
                nt=nt+1
            else:
                tmpthetas[i-nt]= [i,anglebt]
        nn=(n)-nt
        tmpthetas=tmpthetas[:nn]
        thetas=tmpthetas[np.argsort(tmpthetas[:, 1])]



        tmpTukeyDepth=10000000000
        minPoints = 1000000000000000000000
        minId=-1
        # de uno por que el primero (cero) es el que se busca medir
        for i in range(0,nn):
            #minPoints = 1000000000000000000000
            min_angle = thetas[i][1];
            max_angle = (min_angle + (math.pi)) % (2 * math.pi)
            pos_ini = i
            pos_end=len([k for k in thetas if k[1] <= max_angle])-1


            #Cuando se pasa pi se invierten los angulos menores y mayores para el conteo
            if min_angle>max_angle :
                tmp_angle=min_angle
                min_angle=max_angle
                max_angle=tmp_angle
                pos_ini = len([k for k in thetas if k[1] <= min_angle])
                pos_end=i

            if abs(pos_ini- pos_end)+1 <= minPoints:
                if (abs(pos_ini- pos_end)+1 < minPoints):
                    resp = []
                minPoints=abs(pos_ini- pos_end)+1
                minId=thetas[i][0]
                resp.append([minPoints, minId])

            if  (nn - (abs(pos_ini- pos_end)+1))+1 <= minPoints:
                if ((nn - (abs(pos_ini- pos_end)+1))+1 < minPoints):
                    resp = []
                minPoints=(nn - (abs(pos_ini- pos_end)+1))+1
                minId = thetas[i][0]
                resp.append([minPoints, minId])




        return resp


    def MaxTukeyDepth(points, px):
        eps=.000001
        n=len(points)
        tmpthetas=np.empty([n,2])
        resp=[]
        nt=0

        for i in range(0,n):
            anglebt=UTILS.AngleBetweenPoints(px,points[i])
            if(anglebt<=eps):
                nt=nt+1
            else:
                tmpthetas[i-nt]= [i,anglebt]
        nn=(n)-nt
        tmpthetas=tmpthetas[:nn]
        thetas=tmpthetas[np.argsort(tmpthetas[:, 1])]



        maxTukey=-10000

        # de uno por que el primero (cero) es el que se busca medir
        for i in range(0,nn):
            minPoints = 1000000000000000000000
            min_angle = thetas[i][1];
            max_angle = (min_angle + (math.pi)) % (2 * math.pi)
            pos_ini = i
            pos_end=len([k for k in thetas if k[1] <= max_angle])-1


            #Cuando se pasa pi se invierten los angulos menores y mayores para el conteo
            if min_angle>max_angle :
                tmp_angle=min_angle
                min_angle=max_angle
                max_angle=tmp_angle
                pos_ini = len([k for k in thetas if k[1] <= min_angle])
                pos_end=i

            if abs(pos_ini- pos_end)+1 <= minPoints:
                minPoints=abs(pos_ini- pos_end)+1

            if  (nn - (abs(pos_ini- pos_end)+1))+1 <= minPoints:
                minPoints=(nn - (abs(pos_ini- pos_end)+1))+1

            if (maxTukey<=minPoints):
                if (maxTukey<minPoints):
                    resp=[]
                maxTukey=minPoints
                resp.append([minPoints,int(thetas[i][0]),thetas[i][1]])

        return resp



    #Este es el metodo para sacar H deph y S depth del articulo Rosseeuw 1996
    def TukeDepth2(X,u):
        n=len(X)
        eps=.00000001
        nt=0
        nn=0
        alpha=np.empty(n)

        #Se construye el arreglo ALPHA n
        for i in range(0,n):
            dist=np.linalg.norm(X[i]-u)
            if (dist <= eps):
                nt=nt+1
            else:
                tmpPoint=(X[i]-u)/dist
                alphaAngle=math.atan2(tmpPoint[1],tmpPoint[0])
                if (alphaAngle<0):
                    alphaAngle= alphaAngle+2*np.pi

            alpha[i-nt]=alphaAngle
            if(alpha[i-nt]>= (2*np.pi)-eps): alpha[i-nt]=0
        nn=n-nt
        if(nn <= 1):
           return 0


        #Se ordenan los puntos por angulo nlgn
        alphaN=np.sort(alpha[0:nn])

        #Se checa si los puntos estan en pi
        angle = alphaN[0]-alphaN[nn-1]+2*np.pi
        for i in range(1,nn):
            angle=np.max(angle,(alphaN[i]-alphaN[i-1]))

        if(angle >= math.pi+eps):
            return 0

        #Se normaliza con el angulo mas pequeno ademas se cuentan las que estan en la semi circuenferencia de pi de arriba
        angle= alphaN[0]
        nu=0
        for i in range(0,nn):
            alphaN[i]=alphaN[i]-angle
            if(alphaN[i] <= (np.pi- eps)):
                nu=nu+1

        if(nu >= nn):
            return 0

        ja=0
        jb=0
        alphaK=alphaN[0]
        bethaK=alphaN[nu]-np.pi
        nn2= 2*nn
        nbad=0
        i=nu
        nf=nn
        F= np.empty(n)

        for j in range(0,nn2):
            if(alphaK+eps <= bethaK):
                nf=nf+1
                if(ja <= nn):
                    ja=ja+1
                    alphaK=alphaN[ja]
                else:
                    alphaK= (2*np.pi)+1
            else:
                i=i+1
                if(i == nn):
                    i=0
                    nf=nf-nn
                F[i]=nf
                nbad = nbad +  TUKEY.K((nf - i), 2)
                if(jb <= nn):
                    jb=jb+1
                    if((jb+nu) <= nn):
                        bethaK=alphaN(jb+nu)-np.pi
                    else:
                        bethaK=alphaN(jb+nu-nn)+np.pi
                else:
                    bethaK=(2*np.pi)+1
        nums=TUKEY.K(nn,3)-nbad

        #Computation of numh for halfspace depth
        gi=0
        ja=0
        angle= alphaN[0]
        numh = np.min(F(0), (nn - F[0]))
        for i in range(1,nn):
            if(alphaN[i] <= (angle + eps)):
                ja=ja+1
            else:
                gi=gi+ja
                ja=1
                angle=alphaN[i]
            ki=F[i]-gi
            numh = np.min(numh, np.min(ki, (nn - ki)))

        return numh+nt

    def K(self,m,j):
        if(m < j):
            k=0
        else:
            if(j == 1):
                k=m
            if(j==2):
                k=(m*(m-1))/2
            if(j==3):
                k=(m * (m - 1) * (m - 2)) / 6
        return k

# class testTUKEY:
#     points= np.random.rand(100,2)*1000
#
#     depth= TUKEY()
#     px=np.array([345,500])
#     theta=5.4
#     translationvector=[250,30]
#     scalefactor=3.5
#
#     #firstTks=TUKEY.TukeyDepth(points,px)
#
#     firstTks = TUKEY.MaxTukeyDepthAll(points, px)
#     print('------------')
#     print(px)
#     for i in range(len(firstTks)):
#         print(firstTks[i])
#         firstTks[i][3] = (2 * np.pi) - firstTks[i][3]
#         print(TRANSFORMATIONS.Rotation(px,firstTks[i][3]))
#         #print(points[int(firstTks[i][2])])
#         pointsrot=TRANSFORMATIONS.Rotations(points,firstTks[i][3])
#         print(pointsrot[int(firstTks[i][2])])
#
#     print('------------')
#     print('------------')
#
#     pointsTransformed= TRANSFORMATIONS.Rotations(points[:90],theta)
#     #pointsTransformed=TRANSFORMATIONS.Tranlations(pointsTransformed,translationvector)
#     #pointsTransformed=TRANSFORMATIONS.Scales(pointsTransformed,scalefactor)
#
#     pxTr=TRANSFORMATIONS.Rotation(px,theta)
#     #pxTr=TRANSFORMATIONS.Tranlation(pxTr,translationvector)
#     #pxTr=TRANSFORMATIONS.Scale(pxTr,scalefactor)
#
#     print(pxTr)
#     firstTks2=TUKEY.MaxTukeyDepthAll(pointsTransformed,pxTr)
#     for i in range(len(firstTks2)):
#         print(firstTks2[i])
#         firstTks2[i][3]=(2*np.pi)-firstTks2[i][3]
#         print(TRANSFORMATIONS.Rotation(pxTr, firstTks2[i][3]))
#         #print(pointsRotated[int(firstTks2[i][2])])
#         newpoints=TRANSFORMATIONS.Rotations(pointsTransformed,firstTks2[i][3])
#         print(newpoints[int(firstTks2[i][2])])
#     print('------------')
#
#
#
#     print("exit 0")



