import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

#===================================================================
# This is the parameter we want to optimize -> requires_grad=True
#===================================================================
w = torch.tensor(1.0, requires_grad=True)

#=================================
# Evaluación cálculo de costo
#=================================
y_predicted = w * x
loss = (y_predicted - y)**2
print(loss)

#==========================================
# retropropagación para calcular gradiente
#==========================================
loss.backward()
print(w.grad)

#=======================================
# Nuevos coeficientes
# repetir evaluación y retropropagación
#=======================================
with torch.no_grad():
    w -= 0.01 * w.grad
w.grad.zero_()