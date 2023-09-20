import numpy
import torch
import numpy as np
from numpy import array

#parâmetros de inicialização
qtd_amostras = 5000 #Vocês decidem
beta, gamma = 0.4, 0.2 #Não mudar
n, tau = 18, 20 #Não mudar
x = tau*[0]
x.append(1)
for _ in range(0,qtd_amostras):
    xt = x[-1]+(beta*x[-tau]/(1+x[-tau]**n))-gamma*x[-1]
    x.append(xt)
x=x[tau+1:]

