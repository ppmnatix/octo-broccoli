import numpy
import math
import cmath

class PHIJ:
    def __init__(self, data):
        self.data = data

    def getPhij(vertex,J):
        try:
            complexarray= numpy.array(numpy.zeros(len(vertex)),dtype=complex)
            #Se convierten los valores en R2 con su respectivo complejo.
            for i in range(0, len(vertex)):
                complexarray[i]=(vertex[i][0]+ vertex[i][1]*1j)
            #El valor lambda usando la identidad
            lamda = math.cos((2*math.pi)/len(complexarray)) + math.sin((2*math.pi)/len(complexarray))*1j
            dividend=complex(0,0)
            divisor=complex(0,0)
            #Sumatorias por arriba y por abajo del circulo de radio 1 en el plano complejo
            for i in range(1,len(complexarray)+1):
                dividend=dividend+ (pow(lamda,J*i) * complexarray[i-1])
                divisor=divisor+ (pow(lamda,-J*i) * complexarray[i-1])

            cmpDivide=(dividend/divisor)

            r,the=cmath.polar(cmpDivide)
            if(math.isnan(r) or r==0):
                return None

            return cmpDivide
        except ValueError:
            return None


    def getSimplePhij(vertex, J):
        try:
            complexarray = numpy.array(numpy.zeros(len(vertex)), dtype=complex)
            # Se convierten los valores en R2 con su respectivo complejo.
            for i in range(0, len(vertex)):
                complexarray[i] = (vertex[i][0] + vertex[i][1] * 1j)
            # El valor lambda usando la identidad
            lamda = math.cos((2 * math.pi) / len(complexarray)) + math.sin((2 * math.pi) / len(complexarray)) * 1j
            sum = complex(0, 0)
            # Sumatorias por arriba y por abajo del circulo de radio 1 en el plano complejo
            for i in range(1, len(complexarray) + 1):
                sum = sum + (pow(lamda, J * i) * complexarray[i - 1])


            r, the = cmath.polar(sum)
            if (math.isnan(r) or r == 0):
                return None

            return sum
        except ValueError:
            return None
