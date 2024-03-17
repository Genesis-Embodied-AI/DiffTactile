import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

input_size = 2* 136 
hidden_size = 128
num_layers = 2
num_classes = 2
model = RNNModel(input_size, hidden_size, num_layers, num_classes)



##  Deploy here

model_state = torch.load('rnn_model.pth')
model.load_state_dict(model_state)


def predict(model, new_data):
    model.eval()
    L, _, _ = new_data.shape
    new_data = new_data.reshape(L, -1)
    
    with torch.no_grad():
        new_data_tensor = torch.tensor(new_data, dtype=torch.float32).unsqueeze(0)  
        output = model(new_data_tensor)
        predicted = torch.argmax(output, dim=1)
        return predicted.item()
    

result = predict(model, offset_array)
print(f"The result is {result}")







