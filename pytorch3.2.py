import torch

#===============================================
# Calcular el gradiente de forma automática
#===============================================

#==========================================
#Regresión lineal
# f = W *X
#==========================================

#=======================
# ejemplo : f = 2*x
#=======================
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#===================
# modelo
#===================
def forward(x):
    return w * x

#========================
# error: loss = MSE
#========================
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

print(f'Predicción antes del aprendizaje: f(5) = {forward(5).item():.3f}')

#===================
# aprendizaje
#===================
learning_rate = 0.01
n_iters = 100

#===============================
for epoch in range(n_iters):
    # predict = forward pass
    y_pred = forward(X)
    # loss
    l = loss(Y, y_pred)
    # calculate gradients = backward pass
    l.backward()
    # update weights
    #w.data = w.data - learning_rate * w.grad
    with torch.no_grad():
        w -= learning_rate * w.grad
    # zero the gradients after updating
    w.grad.zero_()
    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w.item():.3f}, loss = {l.item():.8f}')
#==================================

print(f'Predicción después del aprendizaje: f(5) = {forward(5).item():.3f}')