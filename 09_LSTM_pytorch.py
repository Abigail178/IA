import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.optim import Adam 
import lightning as L 
from torch.utils.data import TensorDataset, DataLoader 

#=====================================
# Red LSTM con Lightning
#=====================================
class LightningLSTM(L.LightningModule):

    def __init__(self): 
        super().__init__()
        L.seed_everything(seed=42)
    
        self.lstm = nn.LSTM(input_size=1, hidden_size=1) 
         
    #==================================
    # Evaluacion de la red neuronal
    #==================================
    def forward(self, input):
        input_trans = input.view(len(input), 1)
        lstm_out, temp = self.lstm(input_trans)
        prediction = lstm_out[-1] 
        return prediction
        
        
    #==================================
    # Método de descenso de gradientes
    #==================================
    def configure_optimizers(self): 
        return Adam(self.parameters(), lr=0.1) 

    
    #=============================
    # Paso de entrenamiento
    #=============================
    def training_step(self, batch, batch_idx): 
        input_i, label_i = batch
        output_i = self.forward(input_i[0]) 
        loss = (output_i - label_i)**2
        return loss
    
#=================================================
# Crear, entrenar y obtener resultados de la red
#=================================================
model = LightningLSTM() 
    
print("\nNow let's compare the observed and predicted values...")

print("Company A: Observed = 0, Predicted =", 
      model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
print("Company B: Observed = 1, Predicted =", 
      model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

trainer = L.Trainer(max_epochs=300, log_every_n_steps=2)
trainer.fit(model, train_dataloaders=dataloader)

print("After optimization, the parameters are...")
for name, param in model.named_parameters():
    print(name, param.data)

print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())