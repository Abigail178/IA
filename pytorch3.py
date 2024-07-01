import numpy as np 

#================================
# Calcular manualmente
#================================

#=================================
# Regresión lineal
# f = w * x
#=================================

#============================
# ejemplo: f = 2 * x
#============================
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

#===================
# modelo
#===================
def forward(x):
    return w * x

#=======================
# error: loss = MSE
#=======================
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

#================================
# J = MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y)
#================================
def gradient(x, y, y_pred):
    return np.mean(2*x, y_pred - y)

print(f'Prediccion previa al aprendizaje: f(5) = {forward(5):.3f}')

#======================
# aprendizaje
#======================
learning_rate = 0.01
n_iters = 20

#=====================================
for epoch in range(n_iters):
    # predict = forward pass
    y_pred = forward(X)
    # loss
    l = loss(Y, y_pred)
    # calculate gradients
    dw = gradient(X, Y, y_pred)
    # update weights
    w -= learning_rate * dw
    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediccion con aprendizaje completo: f(5) = {forward(5):.3f}')