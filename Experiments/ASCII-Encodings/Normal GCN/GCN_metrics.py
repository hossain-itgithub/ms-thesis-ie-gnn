import os
import pytesseract
from PIL import Image
import json
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.conv5 = GCNConv(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        return F.log_softmax(x, dim=1)

# Download required NLTK data
nltk.download('punkt')

# Utility functions
def extract_text_and_positions_from_image(image_path):
    try:
        img = Image.open(image_path)
        width, height = img.size
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        return data, width, height
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        return None, None, None

def tokenize_text_with_positions(data, width, height):
    tokens, positions = [], []
    num_boxes = len(data['level'])
    for i in range(num_boxes):
        if int(data['conf'][i]) > 0:  # Consider only confident predictions
            text = data['text'][i]
            if text.strip():  # Ignore empty strings
                tokens.extend(word_tokenize(text))
                norm_left = data['left'][i] / width
                norm_top = data['top'][i] / height
                positions.extend([(norm_left, norm_top)]*len(word_tokenize(text)))
    return tokens, positions

def align_labels_with_tokens(tokens, positions, ground_truth):
    labels = []
    for token, pos in zip(tokens, positions):
        label = 4  # Default to "Other"
        for key, value in ground_truth.items():
            if token in value:
                if key == "company":
                    label = 0
                elif key == "date":
                    label = 1
                elif key == "address":
                    label = 2
                elif key == "total":
                    label = 3
                break
        labels.append(label)
    return labels

def create_graph_input(tokens, positions, labels=None, max_len=10):
    padded_tokens = [token.ljust(max_len)[:max_len] for token in tokens]
    node_features = np.array([[ord(char) for char in token] + [positions[i][0], positions[i][1]] for i, token in enumerate(padded_tokens)], dtype=np.float32)
    edge_index = []
    for i in range(len(tokens) - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    edge_index = np.array(edge_index, dtype=np.int64).T
    if labels is not None:
        return node_features, edge_index, labels
    return node_features, edge_index

class InvoiceDataset(Dataset):
    def __init__(self, img_folder, entities_folder=None):
        self.img_folder = img_folder
        self.entities_folder = entities_folder
        self.image_files = [f for f in os.listdir(img_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_folder, self.image_files[idx])
        if self.entities_folder:
            annotation_path = os.path.join(self.entities_folder, self.image_files[idx].replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt'))
        data, width, height = extract_text_and_positions_from_image(image_path)
        if data:
            tokens, positions = tokenize_text_with_positions(data, width, height)
            if tokens and positions:
                if self.entities_folder:
                    with open(annotation_path, 'r') as f:
                        ground_truth = json.load(f)
                    labels = align_labels_with_tokens(tokens, positions, ground_truth)
                    node_features, edge_index, labels = create_graph_input(tokens, positions, labels)
                    graph_data = Data(x=torch.tensor(node_features, dtype=torch.float), edge_index=torch.tensor(edge_index, dtype=torch.long), y=torch.tensor(labels, dtype=torch.long))
                    return graph_data, tokens, labels
                else:
                    node_features, edge_index = create_graph_input(tokens, positions)
                    graph_data = Data(x=torch.tensor(node_features, dtype=torch.float), edge_index=torch.tensor(edge_index, dtype=torch.long))
                    return graph_data, tokens
        return None, None, None

def collate_fn(batch):
    batch = [data for data in batch if data[0] is not None]
    if len(batch) == 0:
        # Return a dummy batch if the batch is empty
        return Batch(x=torch.tensor([], dtype=torch.float), edge_index=torch.tensor([[], []], dtype=torch.long), y=torch.tensor([], dtype=torch.long), batch=torch.tensor([], dtype=torch.long), ptr=torch.tensor([0], dtype=torch.long)), [], []
    if len(batch[0]) == 3:
        data, tokens, labels = zip(*batch)
        return Batch.from_data_list(data), tokens, labels
    else:
        data, tokens = zip(*batch)
        return Batch.from_data_list(data), tokens, None

def test_model(model, dataloader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch, tokens, labels in dataloader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1).cpu().numpy()
            ptr = batch.ptr.cpu().numpy()
            for i in range(len(ptr) - 1):
                predictions.extend(pred[ptr[i]:ptr[i + 1]])
                if labels:
                    true_labels.extend(labels[i])
    return predictions, true_labels

def decode_predictions(predictions, true_labels):
    labels = {0: "company", 1: "date", 2: "address", 3: "total", 4: "other"}
    decoded_predictions = [labels[pred] for pred in predictions]
    decoded_true_labels = [labels[true] for true in true_labels]
    return decoded_predictions, decoded_true_labels

# Load the trained model
model = GCN(input_dim=12, hidden_dim=16, output_dim=5).to(device)
model.load_state_dict(torch.load('gcn_final.pth'))

# Example usage
test_img_folder = 'data3/test/img'
test_entities_folder = 'data3/test/entities'
test_dataset = InvoiceDataset(test_img_folder, test_entities_folder)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

# Test the model and get predictions
predictions, true_labels = test_model(model, test_dataloader)
decoded_predictions, decoded_true_labels = decode_predictions(predictions, true_labels)

# Calculate metrics
print("Classification Report:\n", classification_report(true_labels, predictions, target_names=["company", "date", "address", "total", "other"]))
f1 = f1_score(true_labels, predictions, average='weighted')
precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted')

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
