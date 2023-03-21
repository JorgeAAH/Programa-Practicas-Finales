import math
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy import linalg

#Constantes de la simulación.
M = 1.989e30
s = 0.95
c = 2.998e8
G = 6.674e-11

def metric(r, theta):
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    a = s*G*M/(c**2)
    sigma = (r**2)+((a*costheta)**2)
    delta = (r**2)+(a**2)-2*G*M*r/(c**2)
    gtt = -(delta-((a*sintheta)**2))*(c**2)/sigma
    gtphi = (delta-(r**2)-(a**2))*a*(sintheta**2)*c/sigma
    gphiphi = ((((r**2)+(a**2))**2)-delta*((a*sintheta)**2))*(sintheta**2)/sigma
    grr = sigma/delta
    gthetatheta = sigma
    return [[gtt, 0, 0, gtphi], [0, grr, 0, 0], [0, 0, gthetatheta, 0], [gtphi, 0, 0, gphiphi]]

def metric_inv(r, theta):
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    a = s*G*M/(c**2)
    sigma = (r**2)+((a*costheta)**2)
    delta = (r**2)+(a**2)-2*G*M*r/(c**2)
    gtt = -(delta-((a*sintheta)**2))*(c**2)/sigma
    gtphi = (delta-(r**2)-(a**2))*a*(sintheta**2)*c/sigma
    gphiphi = ((((r**2)+(a**2))**2)-delta*((a*sintheta)**2))*(sintheta**2)/sigma
    grr = sigma/delta
    gthetatheta = sigma
    #Calculo de la metrica inversa.
    det = gtt*gphiphi-(gtphi**2)
    ginv = [[gphiphi/det,0,0,-gtphi/det],[0,1/grr,0,0],[0,0,1/gthetatheta,0],[-gtphi/det,0,0,gtt/det]]
    return ginv

class NumeroDualDosVariables():
    def __init__(self, numero_parametro, primeras_derivadas_parametro):
        self.numero = numero_parametro
        self.primeras_derivadas = primeras_derivadas_parametro
    def __add__(self, other):
        numero_retorno = self.numero + other.numero
        primeras_derivadas_retorno = [self.primeras_derivadas[0] + other.primeras_derivadas[0], self.primeras_derivadas[1] + other.primeras_derivadas[1]]
        numeroDDV_retorno = NumeroDualDosVariables(numero_retorno, primeras_derivadas_retorno)
        return numeroDDV_retorno
    def __sub__(self, other):
        numero_retorno = self.numero - other.numero
        primeras_derivadas_retorno = [self.primeras_derivadas[0] - other.primeras_derivadas[0], self.primeras_derivadas[1] - other.primeras_derivadas[1]]
        numeroDDV_retorno = NumeroDualDosVariables(numero_retorno, primeras_derivadas_retorno)
        return numeroDDV_retorno
    def __mul__(self, other):
        numero_retorno = self.numero*other.numero
        primera_derivada_0_retorno = self.numero*other.primeras_derivadas[0]+self.primeras_derivadas[0]*other.numero
        primera_derivada_1_retorno = self.numero*other.primeras_derivadas[1]+self.primeras_derivadas[1]*other.numero
        numeroDDV_retorno = NumeroDualDosVariables(numero_retorno, [primera_derivada_0_retorno, primera_derivada_1_retorno])
        return numeroDDV_retorno
    def __truediv__(self, other):
        numero_retorno = self.numero/other.numero
        denominador = other.numero*other.numero
        primera_derivada_0_retorno = (self.primeras_derivadas[0]*other.numero-self.numero*other.primeras_derivadas[0])/denominador
        primera_derivada_1_retorno = (self.primeras_derivadas[1]*other.numero-self.numero*other.primeras_derivadas[1])/denominador
        numeroDDV_retorno = NumeroDualDosVariables(numero_retorno, [primera_derivada_0_retorno, primera_derivada_1_retorno])
        return numeroDDV_retorno
    def __pow__(self, other):
        numero_retorno = self.numero**other
        primera_derivada_0_retorno = other*(self.numero**(other-1))*self.primeras_derivadas[0]
        primera_derivada_1_retorno = other*(self.numero**(other-1))*self.primeras_derivadas[1]
        numeroDDV_retorno = NumeroDualDosVariables(numero_retorno, [primera_derivada_0_retorno, primera_derivada_1_retorno])
        return numeroDDV_retorno
    def __neg__(self):
        numeroDDV_retorno = NumeroDualDosVariables(-self.numero, [-self.primeras_derivadas[0], -self.primeras_derivadas[1]])
        return numeroDDV_retorno
    def sin(self):
        numero_retorno = math.sin(self.numero)
        primera_derivada_0_retorno = math.cos(self.numero)*self.primeras_derivadas[0]
        primera_derivada_1_retorno = math.cos(self.numero)*self.primeras_derivadas[1]
        numeroDDV_retorno = NumeroDualDosVariables(numero_retorno, [primera_derivada_0_retorno, primera_derivada_1_retorno])
        return numeroDDV_retorno
    def cos(self):
        numero_retorno = math.cos(self.numero)
        primera_derivada_0_retorno = -math.sin(self.numero)*self.primeras_derivadas[0]
        primera_derivada_1_retorno = -math.sin(self.numero)*self.primeras_derivadas[1]
        numeroDDV_retorno = NumeroDualDosVariables(numero_retorno, [primera_derivada_0_retorno, primera_derivada_1_retorno])
        return numeroDDV_retorno

#Cálculo de la conexión afín usando numeros duales, ya que es necesario derivar una vez.
def affine(r, theta):
    #r corresponderá a la primera variable y theta a la segunda variable.
    DDVr = NumeroDualDosVariables(r, [1,0])
    DDVtheta = NumeroDualDosVariables(theta, [0,1])
    DDVM = NumeroDualDosVariables(M, [0,0])
    DDVs = NumeroDualDosVariables(s, [0,0])
    DDVc = NumeroDualDosVariables(c, [0,0])
    DDVG = NumeroDualDosVariables(G, [0,0])
    DDV2 = NumeroDualDosVariables(2, [0,0])
    costheta = DDVtheta.cos()
    sintheta = DDVtheta.sin()
    a = DDVs*DDVG*DDVM/(DDVc**2)
    sigma = (DDVr**2)+((a*costheta)**2)
    delta = (DDVr**2)+(a**2)-DDV2*DDVG*DDVM*DDVr/(DDVc**2)
    gtt = -(delta-((a*sintheta)**2))*(DDVc**2)/sigma
    gtphi = (delta-(DDVr**2)-(a**2))*a*(sintheta**2)*DDVc/sigma
    gphiphi = ((((DDVr**2)+(a**2))**2)-delta*((a*sintheta)**2))*(sintheta**2)/sigma
    grr = sigma/delta
    gthetatheta = sigma
    #Desde este punto ya no se utilizan números duales.
    #"Tensor" de derivadas parciales. El primer indice es el de la derivada.
    g_der = [[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], [[gtt.primeras_derivadas[0], 0, 0, gtphi.primeras_derivadas[0]], [0, grr.primeras_derivadas[0], 0, 0], [0, 0, gthetatheta.primeras_derivadas[0], 0], [gtphi.primeras_derivadas[0], 0, 0, gphiphi.primeras_derivadas[0]]], [[gtt.primeras_derivadas[1], 0, 0, gtphi.primeras_derivadas[1]], [0, grr.primeras_derivadas[1], 0, 0], [0, 0, gthetatheta.primeras_derivadas[1], 0], [gtphi.primeras_derivadas[1], 0, 0, gphiphi.primeras_derivadas[1]]], [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]]
    #Calculo de la metrica inversa.
    det = gtt.numero*gphiphi.numero-(gtphi.numero**2)
    ginv = [[gphiphi.numero/det,0,0,-gtphi.numero/det],[0,1/grr.numero,0,0],[0,0,1/gthetatheta.numero,0],[-gtphi.numero/det,0,0,gtt.numero/det]]
    #Calculo de la conexion afin. El primer indice es el superior.
    affine_connection = [ [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] , [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] , [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] , [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] ]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    affine_connection[i][j][k] += 0.5*ginv[i][l]*(g_der[j][l][k]+g_der[k][j][l]-g_der[l][j][k])
    return affine_connection

class NumeroTrialDosVariables():
    def __init__(self, numero_parametro, primeras_derivadas_parametro, segundas_derivadas_parametro):
        self.numero = numero_parametro
        self.primeras_derivadas = primeras_derivadas_parametro
        self.segundas_derivadas = segundas_derivadas_parametro
    def __add__(self, other):
        numero_retorno = self.numero + other.numero
        primeras_derivadas_retorno = [self.primeras_derivadas[0] + other.primeras_derivadas[0], self.primeras_derivadas[1] + other.primeras_derivadas[1]]
        segundas_derivadas_retorno = [self.segundas_derivadas[0] + other.segundas_derivadas[0], self.segundas_derivadas[1] + other.segundas_derivadas[1], self.segundas_derivadas[2] + other.segundas_derivadas[2]]
        numeroTDV_retorno = NumeroTrialDosVariables(numero_retorno, primeras_derivadas_retorno, segundas_derivadas_retorno)
        return numeroTDV_retorno
    def __sub__(self, other):
        numero_retorno = self.numero - other.numero
        primeras_derivadas_retorno = [self.primeras_derivadas[0] - other.primeras_derivadas[0], self.primeras_derivadas[1] - other.primeras_derivadas[1]]
        segundas_derivadas_retorno = [self.segundas_derivadas[0] - other.segundas_derivadas[0], self.segundas_derivadas[1] - other.segundas_derivadas[1], self.segundas_derivadas[2] - other.segundas_derivadas[2]]
        numeroTDV_retorno = NumeroTrialDosVariables(numero_retorno, primeras_derivadas_retorno, segundas_derivadas_retorno)
        return numeroTDV_retorno
    def __mul__(self, other):
        #Se realizarán los cálculos según el cuaderno.
        a = self.numero
        da_dx = self.primeras_derivadas[0]
        da_dy = self.primeras_derivadas[1]
        d2a_dx2 = self.segundas_derivadas[0]
        d2a_dxdy = self.segundas_derivadas[1]
        d2a_dy2 = self.segundas_derivadas[2]
        b = other.numero
        db_dx = other.primeras_derivadas[0]
        db_dy = other.primeras_derivadas[1]
        d2b_dx2 = other.segundas_derivadas[0]
        d2b_dxdy = other.segundas_derivadas[1]
        d2b_dy2 = other.segundas_derivadas[2]
        f = a*b
        df_dx = a*db_dx+da_dx*b
        df_dy = a*db_dy+da_dy*b
        d2f_dx2 = a*d2b_dx2+2*da_dx*db_dx+d2a_dx2*b
        d2f_dxdy = a*d2b_dxdy+da_dx*db_dy+da_dy*db_dx+d2a_dxdy*b
        d2f_dy2 = a*d2b_dy2+2*da_dy*db_dy+d2a_dy2*b
        numeroTDV_retorno = NumeroTrialDosVariables(f, [df_dx, df_dy], [d2f_dx2, d2f_dxdy, d2f_dy2])
        return numeroTDV_retorno
    def __truediv__(self, other):
        #Aquí, se invertirá primero el número y luego se usará la multiplicación normal.
        #Procedimiento general funciones de una sola variable.
        a = other.numero
        da_dx = other.primeras_derivadas[0]
        da_dy = other.primeras_derivadas[1]
        d2a_dx2 = other.segundas_derivadas[0]
        d2a_dxdy = other.segundas_derivadas[1]
        d2a_dy2 = other.segundas_derivadas[2]
        #Propio de la función recíproca.
        f = 1/a
        df_da = -1/(a**2)
        d2f_da2 = 2/(a**3)
        #Procedimiento general funciones de una sola variable.
        df_dx = df_da*da_dx
        df_dy = df_da*da_dy
        d2f_dx2 = d2f_da2*(da_dx**2)+df_da*d2a_dx2
        d2f_dxdy = d2f_da2*da_dx*da_dy+df_da*d2a_dxdy
        d2f_dy2 = d2f_da2*(da_dy**2)+df_da*d2a_dy2
        numeroTDV_reciproco = NumeroTrialDosVariables(f, [df_dx, df_dy], [d2f_dx2, d2f_dxdy, d2f_dy2])
        return self*numeroTDV_reciproco
    def __pow__(self, other):
        #Procedimiento general funciones de una sola variable.
        a = self.numero
        da_dx = self.primeras_derivadas[0]
        da_dy = self.primeras_derivadas[1]
        d2a_dx2 = self.segundas_derivadas[0]
        d2a_dxdy = self.segundas_derivadas[1]
        d2a_dy2 = self.segundas_derivadas[2]
        #Propio de la función potencia
        f = a**other
        df_da = other*(a**(other-1))
        d2f_da2 = other*(other-1)*(a**(other-2))
        #Procedimiento general funciones de una sola variable.
        df_dx = df_da*da_dx
        df_dy = df_da*da_dy
        d2f_dx2 = d2f_da2*(da_dx**2)+df_da*d2a_dx2
        d2f_dxdy = d2f_da2*da_dx*da_dy+df_da*d2a_dxdy
        d2f_dy2 = d2f_da2*(da_dy**2)+df_da*d2a_dy2
        numeroTDV_retorno = NumeroTrialDosVariables(f, [df_dx, df_dy], [d2f_dx2, d2f_dxdy, d2f_dy2])
        return numeroTDV_retorno
    def __neg__(self):
        numeroTDV_retorno = NumeroTrialDosVariables(-self.numero, [-self.primeras_derivadas[0], -self.primeras_derivadas[1]], [-self.segundas_derivadas[0], -self.segundas_derivadas[1], -self.segundas_derivadas[2]])
        return numeroTDV_retorno
    def sin(self):
        #Procedimiento general funciones de una sola variable.
        a = self.numero
        da_dx = self.primeras_derivadas[0]
        da_dy = self.primeras_derivadas[1]
        d2a_dx2 = self.segundas_derivadas[0]
        d2a_dxdy = self.segundas_derivadas[1]
        d2a_dy2 = self.segundas_derivadas[2]
        #Propio de la función seno.
        sina = math.sin(a)
        cosa = math.cos(a)
        f = sina
        df_da = cosa
        d2f_da2 = -sina
        #Procedimiento general funciones de una sola variable.
        df_dx = df_da*da_dx
        df_dy = df_da*da_dy
        d2f_dx2 = d2f_da2*(da_dx**2)+df_da*d2a_dx2
        d2f_dxdy = d2f_da2*da_dx*da_dy+df_da*d2a_dxdy
        d2f_dy2 = d2f_da2*(da_dy**2)+df_da*d2a_dy2
        numeroTDV_retorno = NumeroTrialDosVariables(f, [df_dx, df_dy], [d2f_dx2, d2f_dxdy, d2f_dy2])
        return numeroTDV_retorno
    def cos(self):
        #Procedimiento general funciones de una sola variable.
        a = self.numero
        da_dx = self.primeras_derivadas[0]
        da_dy = self.primeras_derivadas[1]
        d2a_dx2 = self.segundas_derivadas[0]
        d2a_dxdy = self.segundas_derivadas[1]
        d2a_dy2 = self.segundas_derivadas[2]
        #Propio de la función coseno.
        sina = math.sin(a)
        cosa = math.cos(a)
        f = cosa
        df_da = -sina
        d2f_da2 = -cosa
        #Procedimiento general funciones de una sola variable.
        df_dx = df_da*da_dx
        df_dy = df_da*da_dy
        d2f_dx2 = d2f_da2*(da_dx**2)+df_da*d2a_dx2
        d2f_dxdy = d2f_da2*da_dx*da_dy+df_da*d2a_dxdy
        d2f_dy2 = d2f_da2*(da_dy**2)+df_da*d2a_dy2
        numeroTDV_retorno = NumeroTrialDosVariables(f, [df_dx, df_dy], [d2f_dx2, d2f_dxdy, d2f_dy2])
        return numeroTDV_retorno
    def numeroDDV(self):
        numeroDDV_retorno = NumeroDualDosVariables(self.numero, self.primeras_derivadas)
        return numeroDDV_retorno
    def derivada0DDV(self):
        numeroDDV_retorno = NumeroDualDosVariables(self.primeras_derivadas[0], [self.segundas_derivadas[0], self.segundas_derivadas[1]])
        return numeroDDV_retorno
    def derivada1DDV(self):
        numeroDDV_retorno = NumeroDualDosVariables(self.primeras_derivadas[1], [self.segundas_derivadas[1], self.segundas_derivadas[2]])
        return numeroDDV_retorno

#Cálculo de la conexión afín y sus derivadas usando numeros triales, ya que es necesario derivar dos veces.
def affine_and_derivatives(r, theta):
    #r corresponderá a la primera variable y theta a la segunda variable.
    TDVr = NumeroTrialDosVariables(r, [1,0], [0,0,0])
    TDVtheta = NumeroTrialDosVariables(theta, [0,1], [0,0,0])
    TDVM = NumeroTrialDosVariables(M, [0,0], [0,0,0])
    TDVs = NumeroTrialDosVariables(s, [0,0], [0,0,0])
    TDVc = NumeroTrialDosVariables(c, [0,0], [0,0,0])
    TDVG = NumeroTrialDosVariables(G, [0,0], [0,0,0])
    TDV2 = NumeroTrialDosVariables(2, [0,0], [0,0,0])
    costheta = TDVtheta.cos()
    sintheta = TDVtheta.sin()
    a = TDVs*TDVG*TDVM/(TDVc**2)
    sigma = (TDVr**2)+((a*costheta)**2)
    delta = (TDVr**2)+(a**2)-TDV2*TDVG*TDVM*TDVr/(TDVc**2)
    gtt = -(delta-((a*sintheta)**2))*(TDVc**2)/sigma
    gtphi = (delta-(TDVr**2)-(a**2))*a*(sintheta**2)*TDVc/sigma
    gphiphi = ((((TDVr**2)+(a**2))**2)-delta*((a*sintheta)**2))*(sintheta**2)/sigma
    grr = sigma/delta
    gthetatheta = sigma
    #Desde este punto ya no se utilizan números trales.
    #"Tensor" de derivadas parciales. El primer indice es el de la derivada.
    DDV0 = NumeroDualDosVariables(0, [0,0])
    g_der = [[[DDV0,DDV0,DDV0,DDV0],[DDV0,DDV0,DDV0,DDV0],[DDV0,DDV0,DDV0,DDV0],[DDV0,DDV0,DDV0,DDV0]], [[gtt.derivada0DDV(), DDV0, DDV0, gtphi.derivada0DDV()], [DDV0, grr.derivada0DDV(), DDV0, DDV0], [DDV0, DDV0, gthetatheta.derivada0DDV(), DDV0], [gtphi.derivada0DDV(), DDV0, DDV0, gphiphi.derivada0DDV()]], [[gtt.derivada1DDV(), DDV0, DDV0, gtphi.derivada1DDV()], [DDV0, grr.derivada1DDV(), DDV0, DDV0], [DDV0, DDV0, gthetatheta.derivada1DDV(), DDV0], [gtphi.derivada1DDV(), DDV0, DDV0, gphiphi.derivada1DDV()]], [[DDV0,DDV0,DDV0,DDV0],[DDV0,DDV0,DDV0,DDV0],[DDV0,DDV0,DDV0,DDV0],[DDV0,DDV0,DDV0,DDV0]]]
    #Calculo de la metrica inversa.
    DDV1 = NumeroDualDosVariables(1, [0,0])
    det = gtt.numeroDDV()*gphiphi.numeroDDV()-(gtphi.numeroDDV()**2)
    ginv = [[gphiphi.numeroDDV()/det,DDV0,DDV0,-gtphi.numeroDDV()/det],[DDV0,DDV1/grr.numeroDDV(),DDV0,DDV0],[DDV0,DDV0,DDV1/gthetatheta.numeroDDV(),DDV0],[-gtphi.numeroDDV()/det,DDV0,DDV0,gtt.numeroDDV()/det]]
    #Calculo de la conexion afin. El primer indice es el superior.
    affine_connection = []
    for i in range(4):
        temp_b = []
        for j in range(4):
            temp_a = []
            for k in range(4):
                temp_a.append(NumeroDualDosVariables(0, [0,0]))
            temp_b.append(temp_a)
        affine_connection.append(temp_b)
    DDV05 = NumeroDualDosVariables(0.5, [0,0])
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    affine_connection[i][j][k] = affine_connection[i][j][k] + DDV05*ginv[i][l]*(g_der[j][l][k]+g_der[k][j][l]-g_der[l][j][k])
    valor_conexion = [ [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] , [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] , [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] , [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] ]
    derivada_r_conexion = [ [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] , [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] , [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] , [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] ]
    derivada_theta_conexion = [ [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] , [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] , [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] , [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] ]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                valor_conexion[i][j][k] = affine_connection[i][j][k].numero
                derivada_r_conexion[i][j][k] = affine_connection[i][j][k].primeras_derivadas[0]
                derivada_theta_conexion[i][j][k] = affine_connection[i][j][k].primeras_derivadas[1]
    return [valor_conexion, derivada_r_conexion, derivada_theta_conexion]

#El estado estará dado por dr/dt, dtheta/dt, dphi/dt, r, theta y phi.
#A partir de eso se calculará d^2r/dt^2, d^2theta/dt^2, d^2phi/dt^2, dr/dt, dtheta/dt y dphi/dt.
def func(t, estado):
    #Definición de dx/dt.
    dxdt = [1.0, estado[0], estado[1], estado[2]]
    #Cálculo de la conexión afín con (r, theta).
    conexion = affine(estado[3], estado[4])
    #Cálculo del término repetitivo.
    sumat = 0.0
    for i in range(4):
        for j in range(4):
            sumat += conexion[0][i][j] * dxdt[i] * dxdt[j]
    #Cálculo de d^2r/dt^2.
    suma = 0.0
    for i in range(4):
        for j in range(4):
            suma += conexion[1][i][j] * dxdt[i] * dxdt[j]
    d2rdt2 = estado[0] * sumat - suma
    #Cálculo de d^2theta/dt^2.
    suma = 0.0
    for i in range(4):
        for j in range(4):
            suma += conexion[2][i][j] * dxdt[i] * dxdt[j]
    d2thetadt2 = estado[1] * sumat - suma
    #Cálculo de d^2phi/dt^2.
    suma = 0.0
    for i in range(4):
        for j in range(4):
            suma += conexion[3][i][j] * dxdt[i] * dxdt[j]
    d2phidt2 = estado[2] * sumat - suma
    return [d2rdt2, d2thetadt2, d2phidt2, estado[0], estado[1], estado[2]]

#La siguiente función calcula la geodesica de una partícula junto con el espín autotransportado por esta.
#El estado estará dado por dr/dt, dtheta/dt, dphi/dt, r, theta y phi unido a S_0, S_1, S_2, S_3. (Componentes covariantes)
#A partir de eso se calculará d^2r/dt^2, d^2theta/dt^2, d^2phi/dt^2, dr/dt, dtheta/dt y dphi/dt unido a dS_0/dt, etc...
def evolucion_geodesica_y_espin(t, estado):
    #Definición de dx/dt.
    dxdt = [1.0, estado[0], estado[1], estado[2]]
    #Cálculo de la conexión afín con (r, theta).
    conexion = affine(estado[3], estado[4])
    #Cálculo del término repetitivo.
    sumat = 0.0
    for i in range(4):
        for j in range(4):
            sumat += conexion[0][i][j] * dxdt[i] * dxdt[j]
    #Cálculo de d^2r/dt^2.
    suma = 0.0
    for i in range(4):
        for j in range(4):
            suma += conexion[1][i][j] * dxdt[i] * dxdt[j]
    d2rdt2 = estado[0] * sumat - suma
    #Cálculo de d^2theta/dt^2.
    suma = 0.0
    for i in range(4):
        for j in range(4):
            suma += conexion[2][i][j] * dxdt[i] * dxdt[j]
    d2thetadt2 = estado[1] * sumat - suma
    #Cálculo de d^2phi/dt^2.
    suma = 0.0
    for i in range(4):
        for j in range(4):
            suma += conexion[3][i][j] * dxdt[i] * dxdt[j]
    d2phidt2 = estado[2] * sumat - suma
    #Calculo de las derivadas del espín (componentes covariantes).
    S = [estado[6], estado[7], estado[8], estado[9]]
    dSdt = [0,0,0,0]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                dSdt[i] += conexion[j][i][k]*dxdt[k]*S[j]
    return [d2rdt2, d2thetadt2, d2phidt2, estado[0], estado[1], estado[2], dSdt[0], dSdt[1], dSdt[2], dSdt[3]]

#La siguiente función calcula la geodesica de una partícula,
#junto con los coeficientes de primer orden de tres perturbaciones iniciales posibles.
#El estado estará dado por dr/dt, dtheta/dt, dphi/dt, r, theta y phi
#unido a los respectivos coeficientes de primer orden para las tres perturbaciones
#iniciales posibles (que serán en dr/dt, dtheta/dt y dphidt para el problema
#de encontrar las condiciones iniciales de la geodesica luminica hasta al observador).
#En resumen: dr0dt, dtheta0dt, dphi0dt, r0, theta0, phi0,
#dr1dt_a, dtheta1dt_a, dphi1dt_a, r1_a, theta1_a, phi1_a,
#dr1dt_b, dtheta1dt_b, dphi1dt_b, r1_b, theta1_b, phi1_b,
#dr1dt_c, dtheta1dt_c, dphi1dt_c, r1_c, theta1_c, phi1_c.
def evolucion_geodesica_y_perturbaciones(t, estado):
    dx0dt = [1.0, estado[0], estado[1], estado[2]]
    dx1dt_a = [0.0, estado[6], estado[7], estado[8]]
    dx1dt_b = [0.0, estado[12], estado[13], estado[14]]
    dx1dt_c = [0.0, estado[18], estado[19], estado[20]]
    #Calcularemos primero algunos TÉRMINOS COMUNES:
    conexion_y_derivadas = affine_and_derivatives(estado[3], estado[4])
    #Término de la conexión afín con los dx0dt.
    suma_t = 0
    for i in range(4):
        for j in range(4):
            suma_t += conexion_y_derivadas[0][0][i][j]*dx0dt[i]*dx0dt[j]
    #Términos de la derivada de la conexión afín con los dx0dt.
    suma_t_dr = 0
    suma_t_dtheta = 0
    for i in range(4):
        for j in range(4):
            suma_t_dr += conexion_y_derivadas[1][0][i][j]*dx0dt[i]*dx0dt[j]
            suma_t_dtheta += conexion_y_derivadas[2][0][i][j]*dx0dt[i]*dx0dt[j]
    #Tríadas de términos para las derivadas de la conexión afín con los dx0dt.
    sumas_i_dr = [0,0,0,0] #Cuidado con los índices...
    sumas_i_dtheta = [0,0,0,0] #Cuidado con los índices...
    for i in range(1, 4):
        for j in range(4):
            for k in range(4):
                sumas_i_dr[i] += conexion_y_derivadas[1][i][j][k]*dx0dt[j]*dx0dt[k]
                sumas_i_dtheta[i] += conexion_y_derivadas[2][i][j][k]*dx0dt[j]*dx0dt[k]
    #CÁLCULO DE d2x0dt2:
    #Tríadas de términos para la conexión afín con dx0dt.
    sumas_i = [0,0,0,0] #Cuidado con los índices...
    for i in range(1, 4):
        for j in range(4):
            for k in range(4):
                sumas_i[i] += conexion_y_derivadas[0][i][j][k]*dx0dt[j]*dx0dt[k]
    #Cálculo de las segundas derivadas de los términos no perturbados.
    d2x0dt2 = [0,0,0,0] #Cuidado con los índices...
    for i in range(1,4):
        d2x0dt2[i] = dx0dt[i]*suma_t-sumas_i[i]
    #CÁLCULO DE d2x1dt2_a:
    #Termino común a las tres segundas derivadas "a", de la conexion afín con dx1dt y dx0dt:
    suma_t_0a = 0
    for i in range(4):
        for j in range(4):
            suma_t_0a += conexion_y_derivadas[0][0][j][k]*dx0dt[j]*dx1dt_a[k]
    #Triada de términos para la conexión afín con dx1dt y dx0dt:
    sumas_i_0a = [0,0,0,0] #Cuidado con los índices...
    for i in range(1,4):
        for j in range(4):
            for k in range(4):
                sumas_i_0a[i] += conexion_y_derivadas[0][i][j][k]*dx0dt[j]*dx1dt_a[k] 
    d2x1dt2_a = [0,0,0,0] #Cuidado con los índices...
    for i in range(1,4):
        d2x1dt2_a[i] = dx1dt_a[i]*suma_t + dx0dt[i]*estado[9]*suma_t_dr + dx0dt[i]*estado[10]*suma_t_dtheta + 2*dx0dt[i]*suma_t_0a - estado[9]*sumas_i_dr[i] - estado[10]*sumas_i_dtheta[i] - 2*sumas_i_0a[i]
    #CÁLCULO DE d2x1dt2_b:
    #Termino común a las tres segundas derivadas "b", de la conexion afín con dx1dt y dx0dt:
    suma_t_0b = 0
    for i in range(4):
        for j in range(4):
            suma_t_0b += conexion_y_derivadas[0][0][j][k]*dx0dt[j]*dx1dt_b[k]
    #Triada de términos para la conexión afín con dx1dt y dx0dt:
    sumas_i_0b = [0,0,0,0] #Cuidado con los índices...
    for i in range(1,4):
        for j in range(4):
            for k in range(4):
                sumas_i_0b[i] += conexion_y_derivadas[0][i][j][k]*dx0dt[j]*dx1dt_b[k] 
    d2x1dt2_b = [0,0,0,0] #Cuidado con los índices...
    for i in range(1,4):
        d2x1dt2_b[i] = dx1dt_b[i]*suma_t + dx0dt[i]*estado[15]*suma_t_dr + dx0dt[i]*estado[16]*suma_t_dtheta + 2*dx0dt[i]*suma_t_0b - estado[15]*sumas_i_dr[i] - estado[16]*sumas_i_dtheta[i] - 2*sumas_i_0b[i]
    #CÁLCULO DE d2x1dt2_c:
    #Termino común a las tres segundas derivadas "c", de la conexion afín con dx1dt y dx0dt:
    suma_t_0c = 0
    for i in range(4):
        for j in range(4):
            suma_t_0c += conexion_y_derivadas[0][0][j][k]*dx0dt[j]*dx1dt_c[k]
    #Triada de términos para la conexión afín con dx1dt y dx0dt:
    sumas_i_0c = [0,0,0,0] #Cuidado con los índices...
    for i in range(1,4):
        for j in range(4):
            for k in range(4):
                sumas_i_0c[i] += conexion_y_derivadas[0][i][j][k]*dx0dt[j]*dx1dt_c[k] 
    d2x1dt2_c = [0,0,0,0] #Cuidado con los índices...
    for i in range(1,4):
        d2x1dt2_c[i] = dx1dt_c[i]*suma_t + dx0dt[i]*estado[21]*suma_t_dr + dx0dt[i]*estado[22]*suma_t_dtheta + 2*dx0dt[i]*suma_t_0c - estado[21]*sumas_i_dr[i] - estado[22]*sumas_i_dtheta[i] - 2*sumas_i_0c[i]
    #Ahora, por fin, el retorno:
    return [d2x0dt2[1], d2x0dt2[2], d2x0dt2[3], dx0dt[1], dx0dt[2], dx0dt[3], d2x1dt2_a[1], d2x1dt2_a[2], d2x1dt2_a[3], dx1dt_a[1], dx1dt_a[2], dx1dt_a[3], d2x1dt2_b[1], d2x1dt2_b[2], d2x1dt2_b[3], dx1dt_b[1], dx1dt_b[2], dx1dt_b[3], d2x1dt2_c[1], d2x1dt2_c[2], d2x1dt2_c[3], dx1dt_c[1], dx1dt_c[2], dx1dt_c[3]]

def cantidades_conservadas(drdt, dthetadt, dphidt, r, theta):
    dxdt = [1, drdt, dthetadt , dphidt]
    #Cálculo de la métrica con (r, theta).
    metrica = metric(r, theta)
    dtdtau = 0
    for i in range(4):
        for j in range(4):
            dtdtau += metrica[i][j]*dxdt[i]*dxdt[j]
    dtdtau = c/math.sqrt(- dtdtau)
    u_contra = [dtdtau, drdt * dtdtau, dthetadt * dtdtau, dphidt * dtdtau]
    magnitud_cuadrivelocidad_normalizada = 0
    for i in range(4):
        for j in range(4):
            magnitud_cuadrivelocidad_normalizada += metrica[i][j]*u_contra[i]*u_contra[j]
    magnitud_cuadrivelocidad_normalizada = math.sqrt(- magnitud_cuadrivelocidad_normalizada)
    u_cov = [0, 0, 0, 0]
    for i in range(4):
        for j in range(4):
            u_cov[i] += metrica[i][j]*u_contra[j]
    energia = -u_cov[0]
    momento_angular = u_cov[3]
    constante_carter = u_cov[2]**2 + (math.cos(theta)**2)*(((s**2)*(G**2)*(M**2)/(c**4))*(c**2-(energia**2)/(c**2))+(momento_angular/math.sin(theta))**2)
    return [magnitud_cuadrivelocidad_normalizada, energia, momento_angular, constante_carter]

def coordenadas_polares_a_cartesianas(r, theta, phi):
    x = r*math.sin(theta)*math.cos(phi)
    y = r*math.sin(theta)*math.sin(phi)
    z = r*math.cos(theta)
    return [x,y,z]

#Se calculan las derivadas parciales de r, theta, phi en terminos de x, y, z.
#En el retorno, el primer índice indica la fila y el segundo la columna.
def jacobiano_polares_en_cartesianas(x, y, z):
    r = math.sqrt((x**2)+(y**2)+(z**2))
    fila_1 = [x/r, y/r, z/r]
    den_fila_2 = (r**2)*math.sqrt((x**2)+(y**2))
    fila_2 = [x*z/den_fila_2, y*z/den_fila_2, -((x**2)+(y**2))/den_fila_2]
    fila_3 = [-y/((x**2)+(y**2)), x/((x**2)+(y**2)), 0]
    return [fila_1, fila_2, fila_3]

def jacobiano_cartesianas_en_polares(r, theta, phi):
    fila_1 = [math.sin(theta)*math.cos(phi), r*math.cos(theta)*math.cos(phi), -r*math.sin(theta)*math.sin(phi)]
    fila_2 = [math.sin(theta)*math.sin(phi), r*math.cos(theta)*math.sin(phi), r*math.sin(theta)*math.cos(phi)]
    fila_3 = [math.cos(theta), -r*math.sin(theta), 0]
    return [fila_1, fila_2, fila_3]

#La siguiente función convierte un vector de diferenciales cartesianas en su respectivo vector de diferenciales polares.
#Para eso, es necesario tener las diferenciales cartesianas y las coordenadas del punto de transformación.
def vector_diferenciales_cartesianas_a_polares(dcartesianas, r, theta, phi):
    coordenadas_cartesianas = coordenadas_polares_a_cartesianas(r, theta, phi)
    jacobiano = jacobiano_polares_en_cartesianas(coordenadas_cartesianas[0], coordenadas_cartesianas[1], coordenadas_cartesianas[2])
    dpolares = [0,0,0]
    for i in range(3):
        for j in range(3):
            dpolares[i] += jacobiano[i][j]*dcartesianas[j]
    return dpolares

#La siguiente función convierte un vector de diferenciales polares en su respectivo vector de diferenciales cartesianas.
#Para eso también son necesarias las coordenadas del punto de transformación.
def vector_diferenciales_polares_a_cartesianas(dpolares, r, theta, phi):
    jacobiano = jacobiano_cartesianas_en_polares(r, theta, phi)
    dcartesianas = [0,0,0]
    for i in range(3):
        for j in range(3):
            dcartesianas[i] += jacobiano[i][j]*dpolares[j]
    return dcartesianas

#La siguiente función toma los valores de las variables que definen el salto y
#computa los residuos que deben hacerse igual a cero.
#derivadas_inicio  es un arreglo con los valores de las primeras derivadas al inicio del unico salto.
#tiempos es un arreglo con los tiempos inicial y final del salto.
#tres_coordenadas_iniciales es un arreglo con las 3 coordenadas iniciales.
#dos_coordenadas_finales es un arreglo con theta y phi finales deseados.
def sistema_ecuaciones_geodesica_luminica(derivadas_inicio, tiempos, tres_coordenadas_iniciales, dos_coordenadas_finales):
    #Arreglo donde se retornarán los valores que deben hacerse cero!
    arreglo_retorno = []
    #Realización del saltos y evaluación de los residuos.
    estado_salto = [*derivadas_inicio,*tres_coordenadas_iniciales]
    sol_salto = solve_ivp(func, (tiempos[0], tiempos[1]), estado_salto, t_eval=[tiempos[1]], method='DOP853')
    #Ahora, colocando las coordenadas finales deseadas en una esfera se definen dos vectores tangentes perpendiculares entre sí.
    #Posteriormente, se define el vector de error entre las coordenadas finales deseadas y las resultantes
    #y se descompone este vector entre sus componentes sobre los vectores tangentes perpendiculares entre sí.
    #Estos componentes son los que se intentan minimizar.
    vector_radial = coordenadas_polares_a_cartesianas(1, dos_coordenadas_finales[0], dos_coordenadas_finales[1])
    vector_tangente_1 = [0,0,0]
    #El primer vector tangente se debe definir en toda la esfera.
    if vector_radial[1] > 0.9:
        y = vector_radial[1]
        z = vector_radial[2]
        vector_tangente_1 = [0, z/math.sqrt((z**2)+(y**2)), -y/math.sqrt((z**2)+(y**2))]
    else:
        x = vector_radial[0]
        z = vector_radial[2]
        vector_tangente_1 = [z/math.sqrt((x**2)+(z**2)), 0, -x/math.sqrt((x**2)+(z**2))]
    vector_tangente_2 = [vector_radial[1]*vector_tangente_1[2]-vector_radial[2]*vector_tangente_1[1], -(vector_radial[0]*vector_tangente_1[2]-vector_radial[2]*vector_tangente_1[0]), vector_radial[0]*vector_tangente_1[1]-vector_radial[1]*vector_tangente_1[0]]
    vector_resultante = coordenadas_polares_a_cartesianas(1, sol_salto.y[4][0], sol_salto.y[5][0])
    vector_error = [0,0,0]
    for i in range(3):
        vector_error[i] = vector_radial[i]-vector_resultante[i]
    error_1 = 0
    for i in range(3):
        error_1 += vector_error[i]*vector_tangente_1[i]
    error_2 = 0
    for i in range(3):
        error_2 += vector_error[i]*vector_tangente_2[i]
    #Se intenta lograr que el final del último salto coincida con el theta y phi requeridos.
    arreglo_retorno.append(error_1)
    arreglo_retorno.append(error_2)
    #Se intenta lograr que se cumpla con la condición de geodésica nula.
    geodesica_nula = 0
    dxdt = [1, derivadas_inicio[0], derivadas_inicio[1], derivadas_inicio[2]]
    metrica = metric(tres_coordenadas_iniciales[0], tres_coordenadas_iniciales[1])
    for i in range(4):
        for j in range(4):
            geodesica_nula += metrica[i][j]*dxdt[i]*dxdt[j]
    #Esta línea deja el error adimensional.
    geodesica_nula = geodesica_nula/(c**2) 
    arreglo_retorno.append(geodesica_nula)
    #Notese que en el retorno, los primeros dos elementos son restas entre ángulos y el tercero tiene unidades adimensionales.
    return arreglo_retorno

#La siguiente función toma los valores de las variables que definen el salto y
#computa los residuos que deben hacerse igual a cero. ADEMÁS, computa el jacobiano de
#los residuos en función de las derivadas_inicio.
#derivadas_inicio  es un arreglo con los valores de las primeras derivadas al inicio del unico salto.
#tiempos es un arreglo con los tiempos inicial y final del salto.
#tres_coordenadas_iniciales es un arreglo con las 3 coordenadas iniciales.
#dos_coordenadas_finales es un arreglo con theta y phi finales deseados.
def sistema_ecuaciones_geodesica_luminica_y_jacobiano(derivadas_inicio, tiempos, tres_coordenadas_iniciales, dos_coordenadas_finales):
    #Arreglo donde se retornarán los valores que deben hacerse cero!
    arreglo_retorno = []
    #Realización del saltos y evaluación de los residuos.
    estado_salto = [*derivadas_inicio,*tres_coordenadas_iniciales, 1,0,0,0,0,0, 0,1,0,0,0,0, 0,0,1,0,0,0]
    sol_salto = solve_ivp(evolucion_geodesica_y_perturbaciones, (tiempos[0], tiempos[1]), estado_salto, t_eval=[tiempos[1]], method='DOP853')
    #Ahora, colocando las coordenadas finales deseadas en una esfera se definen dos vectores tangentes perpendiculares entre sí.
    #Posteriormente, se define el vector de error entre las coordenadas finales deseadas y las resultantes
    #y se descompone este vector entre sus componentes sobre los vectores tangentes perpendiculares entre sí.
    #Estos componentes son los que se intentan minimizar.
    vector_radial = coordenadas_polares_a_cartesianas(1, dos_coordenadas_finales[0], dos_coordenadas_finales[1])
    vector_tangente_1 = [0,0,0]
    #El primer vector tangente se debe definir en toda la esfera.
    if vector_radial[1] > 0.9:
        y = vector_radial[1]
        z = vector_radial[2]
        vector_tangente_1 = [0, z/math.sqrt((z**2)+(y**2)), -y/math.sqrt((z**2)+(y**2))]
    else:
        x = vector_radial[0]
        z = vector_radial[2]
        vector_tangente_1 = [z/math.sqrt((x**2)+(z**2)), 0, -x/math.sqrt((x**2)+(z**2))]
    vector_tangente_2 = [vector_radial[1]*vector_tangente_1[2]-vector_radial[2]*vector_tangente_1[1], -(vector_radial[0]*vector_tangente_1[2]-vector_radial[2]*vector_tangente_1[0]), vector_radial[0]*vector_tangente_1[1]-vector_radial[1]*vector_tangente_1[0]]
    #Cálculo del error usando los resultados de la simulación.
    vector_resultante = coordenadas_polares_a_cartesianas(1, sol_salto.y[4][0], sol_salto.y[5][0])
    vector_error = [0,0,0]
    for i in range(3):
        vector_error[i] = vector_radial[i]-vector_resultante[i]
    error_1 = 0
    for i in range(3):
        error_1 += vector_error[i]*vector_tangente_1[i]
    error_2 = 0
    for i in range(3):
        error_2 += vector_error[i]*vector_tangente_2[i]
    #Se intenta lograr que el final del último salto coincida con el theta y phi requeridos.
    arreglo_retorno.append(error_1)
    arreglo_retorno.append(error_2)
    #Calculo de diferenciales de error en función de dr/dt inicial.
    vector_derror_ddrdt = vector_diferenciales_polares_a_cartesianas([0, sol_salto.y[10][0], sol_salto.y[11][0]], 1, sol_salto.y[4][0], sol_salto.y[5][0])
    for i in range(3):
        vector_derror_ddrdt[i] = -vector_derror_ddrdt[i]
    derror1_ddrdt = 0
    for i in range(3):
        derror1_ddrdt += vector_derror_ddrdt[i]*vector_tangente_1[i]
    derror2_ddrdt = 0
    for i in range(3):
        derror2_ddrdt += vector_derror_ddrdt[i]*vector_tangente_2[i]
    #Calculo de diferenciales de error en función de dtheta/dt inicial.
    vector_derror_ddthetadt = vector_diferenciales_polares_a_cartesianas([0, sol_salto.y[16][0], sol_salto.y[17][0]], 1, sol_salto.y[4][0], sol_salto.y[5][0])
    for i in range(3):
        vector_derror_ddthetadt[i] = -vector_derror_ddthetadt[i]
    derror1_ddthetadt = 0
    for i in range(3):
        derror1_ddthetadt += vector_derror_ddthetadt[i]*vector_tangente_1[i]
    derror2_ddthetadt = 0
    for i in range(3):
        derror2_ddthetadt += vector_derror_ddthetadt[i]*vector_tangente_2[i]
    #Calculo de diferenciales de error en función de dphi/dt inicial.
    vector_derror_ddphidt = vector_diferenciales_polares_a_cartesianas([0, sol_salto.y[22][0], sol_salto.y[23][0]], 1, sol_salto.y[4][0], sol_salto.y[5][0])
    for i in range(3):
        vector_derror_ddphidt[i] = -vector_derror_ddphidt[i]
    derror1_ddphidt = 0
    for i in range(3):
        derror1_ddphidt += vector_derror_ddphidt[i]*vector_tangente_1[i]
    derror2_ddphidt = 0
    for i in range(3):
        derror2_ddphidt += vector_derror_ddphidt[i]*vector_tangente_2[i]
    #Se intenta lograr que se cumpla con la condición de geodésica nula.
    geodesica_nula = 0
    dxdt = [1, derivadas_inicio[0], derivadas_inicio[1], derivadas_inicio[2]]
    metrica = metric(tres_coordenadas_iniciales[0], tres_coordenadas_iniciales[1])
    for i in range(4):
        for j in range(4):
            geodesica_nula += metrica[i][j]*dxdt[i]*dxdt[j]
    #Esta línea deja el error adimensional.
    geodesica_nula = geodesica_nula/(c**2) 
    arreglo_retorno.append(geodesica_nula)
    #Se calculan los diferenciales de este error 3 en función del diferencial de drdt.
    ddxdt_ddrdt = [0, 1, 0, 0]
    derror3_ddrdt = 0
    for i in range(4):
        for j in range(4):
            derror3_ddrdt += metrica[i][j]*ddxdt_ddrdt[i]*dxdt[j] + metrica[i][j]*dxdt[i]*ddxdt_ddrdt[j]
    derror3_ddrdt = derror3_ddrdt/(c**2)
    #Se calculan los diferenciales de este error 3 en función del diferencial de dthetadt.
    ddxdt_ddthetadt = [0, 0, 1, 0]
    derror3_ddthetadt = 0
    for i in range(4):
        for j in range(4):
            derror3_ddthetadt += metrica[i][j]*ddxdt_ddthetadt[i]*dxdt[j] + metrica[i][j]*dxdt[i]*ddxdt_ddthetadt[j]
    derror3_ddthetadt = derror3_ddthetadt/(c**2)
    #Se calculan los diferenciales de este error 3 en función del diferencial de dphidt.
    ddxdt_ddphidt = [0, 0, 0, 1]
    derror3_ddphidt = 0
    for i in range(4):
        for j in range(4):
            derror3_ddphidt += metrica[i][j]*ddxdt_ddphidt[i]*dxdt[j] + metrica[i][j]*dxdt[i]*ddxdt_ddphidt[j]
    derror3_ddphidt = derror3_ddphidt/(c**2)
    #Notese que en el vector retorno, los primeros dos elementos son restas entre ángulos y el tercero tiene unidades adimensionales.
    return [arreglo_retorno, [[derror1_ddrdt, derror1_ddthetadt, derror1_ddphidt], [derror2_ddrdt, derror2_ddthetadt, derror2_ddphidt], [derror3_ddrdt, derror3_ddthetadt, derror3_ddphidt]]]

#El objetivo de esta función es encontrar la dirección en la que debe partir una geodésica lumínica desde
#un punto dado para viajar en la dirección del observador.
def geodesica_luminica_hasta_observador(r_inicial, theta_inicial, phi_inicial, theta_final, phi_final, dpolaresdt_aproximada_parametro = None):
    #Esta línea está relacionada con la escala del error en la velocidad de la luz de la geodesica luminica:
    tiempos = [0, 1000*r_inicial/c]
    dpolaresdt_aproximada = dpolaresdt_aproximada_parametro
    if dpolaresdt_aproximada is None:
        dcartesianasdt_aproximada = coordenadas_polares_a_cartesianas(c, theta_final, phi_final)
        #Al calcular las trivelocidades iniciales anteriores se ha omitido la métrica.
        #Ahora, se escriben estas trivelocidades en coordenadas polares, formando la 
        #aproximación de orden cero para la velocidad inicial de la geodesica luminica.
        dpolaresdt_aproximada = vector_diferenciales_cartesianas_a_polares(dcartesianasdt_aproximada, r_inicial, theta_inicial, phi_inicial)
    #Lanzamiento de fsolve para resolver la ecuación con condiciones de frontera.
    solucion = fsolve(sistema_ecuaciones_geodesica_luminica, dpolaresdt_aproximada, args=(tiempos, [r_inicial, theta_inicial, phi_inicial],[theta_final, phi_final]))
    return solucion

#El objetivo de esta función es encontrar la dirección en la que debe partir una geodésica lúminica desde un punto dado para viajar en la dirección del observador.
#ADEMÁS, se desea poder ajustar la tolerancia en el error de esta dirección!!!.
#Esta función utiliza el método de Newton-Raphson.
def geodesica_luminica_hasta_observador_con_tolerancias(r_inicial, theta_inicial, phi_inicial, theta_final, phi_final, dpolaresdt_aproximada_parametro = None):
    #Esta línea está relacionada con la escala del error en la velocidad de la luz de la geodesica luminica:
    tiempos = [0, 1000*r_inicial/c]
    dpolaresdt_aproximada = dpolaresdt_aproximada_parametro
    if dpolaresdt_aproximada is None:
        dcartesianasdt_aproximada = coordenadas_polares_a_cartesianas(c, theta_final, phi_final)
        #Al calcular las trivelocidades iniciales anteriores se ha omitido la métrica.
        #Ahora, se escriben estas trivelocidades en coordenadas polares, formando la 
        #aproximación de orden cero para la velocidad inicial de la geodesica luminica.
        dpolaresdt_aproximada = vector_diferenciales_cartesianas_a_polares(dcartesianasdt_aproximada, r_inicial, theta_inicial, phi_inicial)
    #Manteniendo la razón entre drdt, dthetadt y dphidt se intentará reescalar la primera aproximación.
    #Para eso, es necesario encontrar el factor de escala resolviendo una cuadrática para que la geodesica nula sea igual a cero.
    a_pol = 0
    dxdt = [0, dpolaresdt_aproximada[0], dpolaresdt_aproximada[1], dpolaresdt_aproximada[2]]
    metrica = metric(r_inicial, theta_inicial)
    for i in range(4):
        for j in range(4):
            a_pol += metrica[i][j]*dxdt[i]*dxdt[j]
    b_pol = 0
    for i in range(3):
        b_pol += metrica[0][i+1]*dpolaresdt_aproximada[i]
    b_pol = 2*b_pol
    c_pol = metrica[0][0]
    factor = (-b_pol+math.sqrt(b_pol**2-4*a_pol*c_pol))/(2*a_pol)
    if factor < 0:
        factor = (-b_pol-math.sqrt(b_pol**2-4*a_pol*c_pol))/(2*a_pol)
    for i in range(3):
        dpolaresdt_aproximada[i] = factor*dpolaresdt_aproximada[i]
    #Lanzamiento del ciclo del método de Newton Raphson.
    solucion = [0,0,0]
    for i in range(3):
        solucion[i] = dpolaresdt_aproximada[i]
    continuar_ciclo = True
    while continuar_ciclo:
        error_y_jacobiano = sistema_ecuaciones_geodesica_luminica_y_jacobiano(solucion, tiempos, [r_inicial, theta_inicial, phi_inicial], [theta_final, phi_final])
        delta_solucion = linalg.solve(error_y_jacobiano[1], [-error_y_jacobiano[0][0], -error_y_jacobiano[0][1], -error_y_jacobiano[0][2]])
        for i in range(3):
            solucion[i] += delta_solucion[i]
        #Notar que estas tolerancias son muy elevadas y tal vez no sean suficientes, pero el ciclo se ejecuta una vez más y probablemente la solución mejora muchísimo.
        if (error_y_jacobiano[0][0] < 0.01) and (error_y_jacobiano[0][1] < 0.01) and (error_y_jacobiano[0][2] < 1e-6):
            continuar_ciclo = False
    return solucion

#La siguiente función genera un espín válido para una partícula dada una posición y "dirección inicial".
#r, theta, phi, drdt, dthetadt, dphidt definen el estado de la partícula.
def generador_espin(theta_espin, phi_espin, r, theta, phi, drdt, dthetadt, dphidt):
    #Se calcula el vector u de la partícula con componentes contravariantes.
    dxdt = [1, drdt, dthetadt , dphidt]
    metrica = metric(r, theta)
    dtdtau = 0
    for i in range(4):
        for j in range(4):
            dtdtau += metrica[i][j]*dxdt[i]*dxdt[j]
    dtdtau = c/math.sqrt(- dtdtau)
    u_contra = [dtdtau, drdt * dtdtau, dthetadt * dtdtau, dphidt * dtdtau]
    #Ahora, se calculan los componentes 1, 2 y 3 del vector de espín.
    direccion_espin = coordenadas_polares_a_cartesianas(1, theta_espin, phi_espin)
    componentes_i_espin = vector_diferenciales_cartesianas_a_polares(direccion_espin, r, theta, phi)
    S_cov = [0, componentes_i_espin[0], componentes_i_espin[1], componentes_i_espin[2]]
    for i in range(1,4):
        S_cov[0] -= u_contra[i]*S_cov[i]
    S_cov[0] = S_cov[0]/u_contra[0]
    return S_cov

#La siguiente función calcula el coseno del ángulo, en el marco de referencia estático de una partícula, entre los componentes espaciales de dos cuadrivectores.
#r, theta, phi, drdt, dthetadt, dphidt definen el estado de la partícula.
def coseno_angulo_cuadrivectores(cuadrivector_1, cuadrivector_2, r, theta, phi, drdt, dthetadt, dphidt):
    #Se generan tres vectores de tipo espín (en el marco comóvil sus componentes temporales son iguales a cero). Estos vectores deberan
    #ser ortonormalizados para que permitan calcular los componentes de los cuadrivectores 1 y 2 en el marco de referencia comóvil.
    vector_unitario_1 = generador_espin(math.pi/2, 0, r, theta, phi, drdt, dthetadt, dphidt)
    vector_unitario_2 = generador_espin(math.pi/2, math.pi/2, r, theta, phi, drdt, dthetadt, dphidt)
    vector_unitario_3 = generador_espin(0, 0, r, theta, phi, drdt, dthetadt, dphidt)
    #Normalización del vector 1:
    metrica_inversa = metric_inv(r, theta)
    magnitud = 0
    for i in range(4):
        for j in range(4):
            magnitud += metrica_inversa[i][j]*vector_unitario_1[i]*vector_unitario_1[j]
    magnitud = math.sqrt(magnitud)
    for i in range(4):
        vector_unitario_1[i] = vector_unitario_1[i]/magnitud
    #Ortogonalización del vector 2:
    magnitud_proyeccion12 = 0
    for i in range(4):
        for j in range(4):
            magnitud_proyeccion12 += metrica_inversa[i][j]*vector_unitario_1[i]*vector_unitario_2[j]
    for i in range(4):
        vector_unitario_2[i] = vector_unitario_2[i] - magnitud_proyeccion12*vector_unitario_1[i]
    #Normalizacion del vector 2:
    magnitud = 0
    for i in range(4):
        for j in range(4):
            magnitud += metrica_inversa[i][j]*vector_unitario_2[i]*vector_unitario_2[j]
    magnitud = math.sqrt(magnitud)
    for i in range(4):
        vector_unitario_2[i] = vector_unitario_2[i]/magnitud
    #Ortogonalización del vector 3:
    magnitud_proyeccion13 = 0
    for i in range(4):
        for j in range(4):
            magnitud_proyeccion13 += metrica_inversa[i][j]*vector_unitario_1[i]*vector_unitario_3[j]
    magnitud_proyeccion23 = 0
    for i in range(4):
        for j in range(4):
            magnitud_proyeccion23 += metrica_inversa[i][j]*vector_unitario_2[i]*vector_unitario_3[j]
    for i in range(4):
        vector_unitario_3[i] = vector_unitario_3[i] - magnitud_proyeccion13*vector_unitario_1[i] - magnitud_proyeccion23*vector_unitario_2[i]
    #Normalización del vector 3:
    magnitud = 0
    for i in range(4):
        for j in range(4):
            magnitud += metrica_inversa[i][j]*vector_unitario_3[i]*vector_unitario_3[j]
    magnitud = math.sqrt(magnitud)
    for i in range(4):
        vector_unitario_3[i] = vector_unitario_3[i]/magnitud
    #Ahora, usando los vectores unitarios se calculan los componentes de los cuadrivectores en x, y y z:
    #Cuadrivector 1.
    cuadrivector_1_x = 0
    for i in range(4):
        for j in range(4):
            cuadrivector_1_x += metrica_inversa[i][j]*cuadrivector_1[i]*vector_unitario_1[j]
    cuadrivector_1_y = 0
    for i in range(4):
        for j in range(4):
            cuadrivector_1_y += metrica_inversa[i][j]*cuadrivector_1[i]*vector_unitario_2[j]
    cuadrivector_1_z = 0
    for i in range(4):
        for j in range(4):
            cuadrivector_1_z += metrica_inversa[i][j]*cuadrivector_1[i]*vector_unitario_3[j]
    #Cuadrivector 2.
    cuadrivector_2_x = 0
    for i in range(4):
        for j in range(4):
            cuadrivector_2_x += metrica_inversa[i][j]*cuadrivector_2[i]*vector_unitario_1[j]
    cuadrivector_2_y = 0
    for i in range(4):
        for j in range(4):
            cuadrivector_2_y += metrica_inversa[i][j]*cuadrivector_2[i]*vector_unitario_2[j]
    cuadrivector_2_z = 0
    for i in range(4):
        for j in range(4):
            cuadrivector_2_z += metrica_inversa[i][j]*cuadrivector_2[i]*vector_unitario_3[j]
    #Calculando el coseno del angulo.
    magnitud_espacial_1 = math.sqrt((cuadrivector_1_x**2)+(cuadrivector_1_y**2)+(cuadrivector_1_z**2))
    magnitud_espacial_2 = math.sqrt((cuadrivector_2_x**2)+(cuadrivector_2_y**2)+(cuadrivector_2_z**2))
    producto_punto = cuadrivector_1_x*cuadrivector_2_x + cuadrivector_1_y*cuadrivector_2_y + cuadrivector_1_z*cuadrivector_2_z
    return producto_punto/(magnitud_espacial_1*magnitud_espacial_2)

#Parametros simulación.
r0 = 6e3
theta0 = 1.4
phi0 = 0.0
drdt0 = 0.0
dthetadt0 = 0
dphidt0 = 25000
final_time = 1e-3
step_number = 3000

#Parametros espin
theta0_espin = 0
phi0_espin = 0
S0 = generador_espin(theta0_espin, phi0_espin, r0, theta0, phi0, drdt0, dthetadt0, dphidt0)

#Parametros observacionales.
#theta_observacion = 1
theta_observacion = 0
phi_observacion = 0

#Cálculos intermedios.
time_step = final_time/step_number
time_array = []
for i in range(step_number):
  time_array.append(time_step*(i+0.5))

#Resolviendo ecuación diferencial para trayectoria del agujero negro.
sol = solve_ivp(evolucion_geodesica_y_espin, (0.0, final_time), [drdt0, dthetadt0, dphidt0, r0, theta0, phi0, S0[0], S0[1], S0[2], S0[3]], t_eval=time_array, method='DOP853')

#Cálculo de las derivadas iniciales que definen cada geodésica lumínica.
#Primero se ejecuta el calculo para la geodesica inicial, que servirá para
#iniciar los cálculos con una buena aproximación.
derivadas_polares = geodesica_luminica_hasta_observador_con_tolerancias(r0, theta0, phi0, theta_observacion, phi_observacion)
print(derivadas_polares)
derivadas_cartesianas_ultima_iteracion = vector_diferenciales_polares_a_cartesianas(derivadas_polares, r0, theta0, phi0)
#Se llena el arreglo con las derivadas iniciales.
arreglo_derivadas_iniciales = []
#Y se llena el arreglo con los angulos de observacion.
arreglo_coseno_observacion = []
for i in range(step_number):
    derivadas_polares = vector_diferenciales_cartesianas_a_polares(derivadas_cartesianas_ultima_iteracion, sol.y[3][i], sol.y[4][i], sol.y[5][i])
    derivadas_polares = geodesica_luminica_hasta_observador_con_tolerancias(sol.y[3][i], sol.y[4][i], sol.y[5][i], theta_observacion, phi_observacion, dpolaresdt_aproximada_parametro = derivadas_polares)
    arreglo_derivadas_iniciales.append(derivadas_polares)
    coseno_observacion = coseno_angulo_cuadrivectores([sol.y[6][i], sol.y[7][i], sol.y[8][i], sol.y[9][i]], [1, derivadas_polares[0], derivadas_polares[1], derivadas_polares[2]], sol.y[3][i], sol.y[4][i], sol.y[5][i], sol.y[0][i], sol.y[1][i], sol.y[2][i])
    arreglo_coseno_observacion.append(coseno_observacion)
    derivadas_cartesianas_ultima_iteracion = vector_diferenciales_polares_a_cartesianas(derivadas_polares, sol.y[3][i], sol.y[4][i], sol.y[5][i])
    if 0 == i%100:
        print(i)
        print(derivadas_cartesianas_ultima_iteracion)
        print(derivadas_polares)
        print(coseno_observacion)
print(arreglo_coseno_observacion)

#Calculo de las cantidades conservadas iniciales, usando (drdt, dthetadt, dphidt, r, theta).
cantidades_conservadas_iniciales = cantidades_conservadas(sol.y[0][100], sol.y[1][100], sol.y[2][100], sol.y[3][100], sol.y[4][100])
print("------------------------------")
print("Cantidad conservadas iniciales:")
print("Razon de la magnitud de la cuadrivelocidad con c: " + str(cantidades_conservadas_iniciales[0]))
print("Energia: " + str(cantidades_conservadas_iniciales[1]))
print("Momento angular paralelo al spin: " + str(cantidades_conservadas_iniciales[2]))
print("Constante de Carter: " + str(cantidades_conservadas_iniciales[3]))

#Calculo de las cantidades conservadas finales, usando (drdt, dthetadt, dphidt, r, theta).
cantidades_conservadas_finales = cantidades_conservadas(sol.y[0][1600], sol.y[1][1600], sol.y[2][1600], sol.y[3][1600], sol.y[4][1600])
print("------------------------------")
print("Cantidad conservadas finales:")
print("Razon de la magnitud de la cuadrivelocidad con c: " + str(cantidades_conservadas_finales[0]))
print("Energia: " + str(cantidades_conservadas_finales[1]))
print("Momento angular paralelo al spin: " + str(cantidades_conservadas_finales[2]))
print("Constante de Carter: " + str(cantidades_conservadas_finales[3]))

print("------------------------------")
print("Número de evaluaciones: " + str(sol.nfev))
