from flask import Flask, request, jsonify
import torch
import torch.nn as nn

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

modelo = Casos(3)
modelo.load_state_dict(torch.load('best_model.pth'))
modelo.eval()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    temp_min = data['temp_min']
    temp_max = data['temp_max']
    precipitacao_total = data['precipitacao_total']
    
    features = torch.tensor([[temp_min, temp_max, precipitacao_total]], dtype=torch.float32)
    
    with torch.no_grad():
        predicao = modelo(features)
    
    predicao = predicao.numpy().tolist()
    
    return jsonify({'casos': round(predicao[0][0])})

if __name__ == '__main__':
    app.run(debug=True)