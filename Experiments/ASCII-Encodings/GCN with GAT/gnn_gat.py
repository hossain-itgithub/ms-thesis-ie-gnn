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
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import matplotlib

matplotlib.use('webagg')

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCN_GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN_GAT, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.gat_conv = GATConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.conv5 = GCNConv(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if edge_index.size(1) == 0:  # Skip the batch if there are no edges
            return torch.zeros((x.size(0), 5), device=x.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.gat_conv(x, edge_index)
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
def resize_image(image, target_height=1000):
    width, height = image.size
    aspect_ratio = width / height
    new_height = target_height
    new_width = int(aspect_ratio * new_height)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image, new_width, new_height

def extract_text_and_positions_from_image(image_path):
    try:
        img = Image.open(image_path)
        resized_img, width, height = resize_image(img)
        data = pytesseract.image_to_data(resized_img, output_type=pytesseract.Output.DICT)
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
    # Create a dictionary to store line numbers and corresponding node indices
    line_dict = {}
    node_features = []
    node_indices = []

    for i, token in enumerate(tokens):
        line_number = positions[i][1]  # Use the y-coordinate as the line number
        if line_number not in line_dict:
            line_dict[line_number] = []
        line_dict[line_number].append(i)
        
        padded_token = token.ljust(max_len)[:max_len]
        node_features.append([ord(char) for char in padded_token] + list(positions[i]))
        node_indices.append(i)

    node_features = np.array(node_features, dtype=np.float32)

    # Create edges based on line numbers and adjacency
    edge_index = []
    for line, indices in line_dict.items():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                edge_index.append([indices[i], indices[j]])
                edge_index.append([indices[j], indices[i]])
            if i < len(indices) - 1:
                edge_index.append([indices[i], indices[i + 1]])
                edge_index.append([indices[i + 1], indices[i]])

    edge_index = np.array(edge_index, dtype=np.int64).T

    if labels is not None:
        return node_features, edge_index, labels
    return node_features, edge_index

class InvoiceDataset(Dataset):
    def __init__(self, img_folder, entities_folder=None, ocr_folder=None):
        self.img_folder = img_folder
        self.entities_folder = entities_folder
        self.ocr_folder = ocr_folder
        self.image_files = [f for f in os.listdir(img_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.img_folder, image_file)
        ocr_path = os.path.join(self.ocr_folder, image_file.replace('.png', '.json').replace('.jpg', '.json').replace('.jpeg', '.json'))
        
        if self.entities_folder:
            annotation_path = os.path.join(self.entities_folder, image_file.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt'))
        
        with open(ocr_path, 'r') as f:
            ocr_data = json.load(f)
        
        tokens = ocr_data['tokens']
        positions = ocr_data['positions']
        
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

# Visualization Function
def visualize_graph_and_image(graph_data, tokens, labels, image_path, ocr_text_path):
    # Convert the graph data to a NetworkX graph
    G = to_networkx(graph_data, to_undirected=True)
    pos = {i: (graph_data.x[i, -2].item(), graph_data.x[i, -1].item()) for i in range(graph_data.num_nodes)}

    print(f"Number of nodes: {graph_data.num_nodes}")

    # Load and display the original image
    img = Image.open(image_path)
    plt.figure(figsize=(20, 12))

    # Subplot for the original image
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")

    # Subplot for the graph representation with token labels
    plt.subplot(2, 2, 2)
    nx.draw(G, pos, with_labels=True, labels={i: tokens[i] for i in pos.keys()}, node_size=500, node_color="skyblue", font_size=10, font_weight="bold", alpha=0.9)
    plt.title("Graph Representation")

    # Subplot for ground truth labels
    plt.subplot(2, 2, 3)
    nx.draw(G, pos, with_labels=True, labels={i: labels[i] for i in pos.keys()}, node_size=500, node_color="lightgreen", font_size=10, font_weight="bold", alpha=0.9)
    plt.title("Ground Truth Labels")

    # Subplot for token embeddings
    plt.subplot(2, 2, 4)
    embeddings = graph_data.x[:, :-2].numpy()  # Exclude position info
    plt.imshow(embeddings, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Token Embeddings")

    plt.tight_layout()
    plt.show()

    # Dump OCR text for debugging
    with open(ocr_text_path, 'w') as f:
        for i, token in enumerate(tokens):
            f.write(f"{token}\t{graph_data.x[i, -2].item()}\t{graph_data.x[i, -1].item()}\n")


def decode_predictions(predictions, true_labels):
    labels = {0: "company", 1: "date", 2: "address", 3: "total", 4: "other"}
    decoded_predictions = [labels[pred] for pred in predictions]
    decoded_true_labels = [labels[true] for true in true_labels]
    return decoded_predictions, decoded_true_labels


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

# Example usage
img_folder = 'data3/train/img'
entities_folder = 'data3/train/entities'
ocr_folder = 'data3/train/ocr_texts'

# Define GCN model, optimizer, scheduler, and loss function
input_dim = 12  # 10 for padded tokens + 2 for position
hidden_dim = 1024
output_dim = 5  # 4 labels + 1 for "Other"
batch_size = 8
lr = 0.002
weight_decay = 5e-8
stepLR_gamma = 0.1
stepLR_stepSize = 20




train_dataset = InvoiceDataset(img_folder, entities_folder, ocr_folder)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=16)

dataset_index = 3
graph_data, tokens, labels = train_dataset[dataset_index]  # Replace 0 with any index to visualize different graphs
if tokens is not None and labels is not None:
    tokens = list(tokens)
    labels = list(labels)
    image_path = os.path.join(img_folder, train_dataset.image_files[dataset_index])
    ocr_text_path = "ocr_text_debug.txt"
    visualize_graph_and_image(graph_data, tokens, labels, image_path, ocr_text_path)
else:
    print("Error in generating graph data")


model = GCN_GAT(input_dim, hidden_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=stepLR_stepSize, gamma=stepLR_gamma)
criterion = torch.nn.CrossEntropyLoss()
scaler = GradScaler()  # For mixed precision training

# # Training loop
model.train()
for epoch in range(100):  # Train for 100 epochs
    total_loss = 0
    for batch, tokens, labels in train_dataloader:
        if batch.x.size(0) == 0:  # Check if the batch is empty
            continue
        batch = batch.to(device)
        optimizer.zero_grad()
        with autocast():  # Mixed precision context
            out = model(batch)
            loss = criterion(out, batch.y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    scheduler.step()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}')
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'gcn_gat_epoch_{epoch + 1}.pth')

print("Training complete")
# Save the final model
torch.save(model.state_dict(), 'gcn_gat_final.pth')

# Example usage
test_img_folder = 'data3/test/img'
test_entities_folder = 'data3/test/entities'
test_ocr_folder = 'data3/test/ocr_texts'

test_dataset = InvoiceDataset(test_img_folder, test_entities_folder, test_ocr_folder)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=16)

# Load the trained model
model = GCN_GAT(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
model.load_state_dict(torch.load('gcn_gat_final.pth'))

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
