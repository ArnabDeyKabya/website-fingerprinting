
"""
from flask import Flask, request, jsonify, send_from_directory
# additional imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)

stored_traces = []
stored_heatmaps = []

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/collect_trace', methods=['POST'])
def collect_trace():
    try:
        trace = request.get_json()

        if not isinstance(trace, list) or not all(isinstance(i, int) for i in trace):
            return jsonify({'error': 'Invalid trace format'}), 400

        stored_traces.append(trace)

        # Compute stats
        trace_np = np.array(trace)
        min_val = int(trace_np.min())
        max_val = int(trace_np.max())
        sample_count = len(trace)
        value_range = max_val - min_val

        # Create heatmap image
        fig, ax = plt.subplots()
        ax.imshow([trace], cmap='hot', aspect='auto')
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        heatmap_b64 = base64.b64encode(buf.read()).decode('utf-8')
        stored_heatmaps.append(heatmap_b64)

        return jsonify({
            'message': 'Trace received',
            'heatmap': heatmap_b64,
            'stats': {
                'min': min_val,
                'max': max_val,
                'range': value_range,
                'samples': sample_count
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_results', methods=['POST'])
def clear_results():
    stored_traces.clear()
    stored_heatmaps.clear()
    return jsonify({'message': 'Results cleared successfully'})




# Additional endpoints can be implemented here as needed.

# Optional: return stored traces
@app.route('/api/traces', methods=['GET'])
def get_traces():
    return jsonify(stored_traces)

# Optional: return stored heatmaps
@app.route('/api/heatmaps', methods=['GET'])
def get_heatmaps():
    return jsonify(stored_heatmaps)

@app.route('/api/get_results', methods=['GET'])
def get_results():
    return jsonify({'traces': stored_traces})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)





"""


import os
import json
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

stored_traces = []
stored_heatmaps = []
stored_predictions = []

DATASET_INPUT_SIZE = 1000
MODEL_PATH = 'saved_models/complex_fingerprint_model.pth'
LABELS = [
    'https://cse.buet.ac.bd/moodle/',
    'https://google.com',
    'https://prothomalo.com'
]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ComplexFingerprintClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        conv_output_size = input_size // 8
        self.fc_input_size = conv_output_size * 128
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size*2)
        self.bn4 = nn.BatchNorm1d(hidden_size*2)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(-1, self.fc_input_size)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)

model = ComplexFingerprintClassifier(
    input_size=DATASET_INPUT_SIZE,
    hidden_size=128,
    num_classes=len(LABELS)
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def predict_site(trace_list):
    arr = np.array(trace_list, dtype=float)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    tensor = torch.tensor(arr, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        idx = int(logits.argmax(dim=1).item())
    return LABELS[idx]

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/collect_trace', methods=['POST'])
def collect_trace():
    try:
        trace = request.get_json()
        if not isinstance(trace, list):
            return jsonify({'error': 'Invalid trace format'}), 400

        stored_traces.append(trace)

        arr = np.array(trace)
        min_val = int(arr.min())
        max_val = int(arr.max())
        sample_count = len(trace)
        value_range = max_val - min_val

        fig, ax = plt.subplots()
        ax.imshow([trace], cmap='hot', aspect='auto')
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        heatmap_b64 = base64.b64encode(buf.read()).decode('utf-8')
        stored_heatmaps.append(heatmap_b64)

        prediction = predict_site(trace)
        stored_predictions.append(prediction)

        return jsonify({
            'message': 'Trace received',
            'heatmap': heatmap_b64,
            'stats': {
                'min': min_val,
                'max': max_val,
                'range': value_range,
                'samples': sample_count
            },
            'prediction': prediction
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_results', methods=['POST'])
def clear_results():
    stored_traces.clear()
    stored_heatmaps.clear()
    stored_predictions.clear()
    return jsonify({'message': 'Results cleared successfully'})

@app.route('/api/get_results', methods=['GET'])
def get_results():
    return jsonify({'traces': stored_traces})

@app.route('/api/get_heatmaps', methods=['GET'])
def get_heatmaps():
    return jsonify({'heatmaps': stored_heatmaps})

@app.route('/api/get_predictions', methods=['GET'])
def get_predictions():
    return jsonify({'predictions': stored_predictions})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
