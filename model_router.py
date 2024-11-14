from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

class Casos(nn.Module):
    def __init__(self, input_dim):
        super(Casos, self).__init__()
        self.rede = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.75),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.75),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.7), 
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.rede(x)


input_dim = 3 
model = Casos(input_dim)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    temp_max = data['temp_max']
    temp_min = data['temp_min']
    precipitacao_total = data['precipitacao_total']
    
    input_data = pd.DataFrame([[temp_max, temp_min, precipitacao_total]], columns=['temp_max', 'temp_min', 'precipitacao_total'])
    
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    
    input_tensor = torch.FloatTensor(input_data_scaled)
    
    with torch.no_grad():
        prediction = model(input_tensor)
    
    return jsonify({'casos': round(prediction.item())})

if __name__ == '__main__':
    app.run(debug=True)