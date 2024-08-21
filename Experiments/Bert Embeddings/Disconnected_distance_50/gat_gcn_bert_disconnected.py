import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data, Dataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard SummaryWriter
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from PIL import Image
from torchinfo import summary
import matplotlib

matplotlib.use('webagg')

label_dict = {"company": 0, "date": 1, "address": 2, "total": 3, "other": 4}

# Define paths
img_folder = "data3/train/img"
ocr_folder = "data3/train/ocr_texts"
label_folder = "data3/train/entities"
emb_folder = "data3/train/embeddings"
test_img_folder = "data3/test/img"
test_ocr_folder = "data3/test/ocr_texts"
test_label_folder = "data3/test/entities"
test_emb_folder = "data3/test/embeddings"
model_save_path = "gat_gcn_model.pth"
logs_folder = "logs"

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Create logs folder if it doesn't exist
os.makedirs(logs_folder, exist_ok=True)

# Load pretrained multilingual BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
bert_model = AutoModel.from_pretrained("bert-base-multilingual-cased").to(device)

# Function to get text embeddings using BERT
def get_text_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Get the mean of the output embeddings

# Function to generate learnable positional embeddings
class PositionalEmbedding(nn.Module):
    def __init__(self, emb_size):
        super(PositionalEmbedding, self).__init__()
        self.linear = nn.Linear(4, emb_size)  # 4 for bounding box coordinates [x1, y1, x2, y2]

    def forward(self, bounding_boxes):
        return self.linear(bounding_boxes)

# Combine text embeddings, positional embeddings, and confidence scores into node features
class NodeFeatureCombiner(nn.Module):
    def __init__(self, text_emb_size, pos_emb_size):
        super(NodeFeatureCombiner, self).__init__()
        self.fc = nn.Linear(text_emb_size + pos_emb_size + 1, text_emb_size + pos_emb_size + 1)

    def forward(self, text_emb, pos_emb, conf_score):
        combined = torch.cat((text_emb, pos_emb, conf_score), dim=1)
        return self.fc(combined)

# Spatial Transformer Network (STN) for normalizing bounding box coordinates
class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        self.localization = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, 4)
        )

    def forward(self, bbox):
        theta = self.localization(bbox)
        return theta

# Graph Convolutional and Attention Network (GCNGAT)
class GCNGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_gcn_layers, num_gat_layers, heads=8, dropout=0.6):
        super(GCNGAT, self).__init__()
        self.gcn_layers = nn.ModuleList()
        self.gat_layers = nn.ModuleList()

        # Input GCN layer
        self.gcn_layers.append(GCNConv(in_channels, hidden_channels))

        # Hidden GCN layers
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))

        # Input GAT layer
        self.gat_layers.append(GATConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout))

        # Hidden GAT layers
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))

        # Output GAT layer
        self.gat_layers.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index)
            x = F.elu(x)
        
        for gat_layer in self.gat_layers[:-1]:
            x = gat_layer(x, edge_index)
            x = F.elu(x)

        x = self.gat_layers[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

# Custom Dataset
class OCRDataset(Dataset):
    def __init__(self, img_folder, ocr_folder, label_folder, emb_folder, generate_embeddings=True, distance_threshold=50):
        self.img_folder = img_folder
        self.ocr_folder = ocr_folder
        self.label_folder = label_folder
        self.emb_folder = emb_folder
        self.generate_embeddings = generate_embeddings
        self.file_names = [f.split(".")[0] for f in os.listdir(ocr_folder) if f.endswith(".json")]
        self.distance_threshold = distance_threshold

    def __len__(self):
        return len(self.file_names)

    def get_embeddings(self, text, file_name):
        emb_path = os.path.join(self.emb_folder, file_name + ".pt")
        if os.path.exists(emb_path):
            embeddings = torch.load(emb_path)
        else:
            embeddings = get_text_embeddings(text)
            torch.save(embeddings, emb_path)
        return embeddings

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        ocr_path = os.path.join(self.ocr_folder, file_name + ".json")
        label_path = os.path.join(self.label_folder, file_name + ".txt")

        if not os.path.exists(ocr_path) or not os.path.exists(label_path):
            print(f"Missing OCR or label data for file: {file_name}")
            return None

        with open(ocr_path, "r") as ocr_file:
            ocr_data = json.load(ocr_file)
        
        with open(label_path, "r") as label_file:
            labels = json.load(label_file)

        # Group words by line and concatenate
        lines = {}
        for item in ocr_data:
            line_num = item['bbox'][1]  # Using the top coordinate as a simple line identifier
            if line_num not in lines:
                lines[line_num] = []
            lines[line_num].append(item)

        words = []
        confidences = []
        bounding_boxes = []
        for line in lines.values():
            line_text = ' '.join([word['word'] for word in line])
            words.append(line_text)
            confidences.append(sum([word['confidence'] for word in line]) / len(line))
            # Compute a bounding box for the entire line
            x1 = min([word['bbox'][0] for word in line])
            y1 = min([word['bbox'][1] for word in line])
            x2 = max([word['bbox'][2] for word in line])
            y2 = max([word['bbox'][3] for word in line])
            bounding_boxes.append([x1, y1, x2, y2])

        if not words:
            print(f"No words found in OCR data for file: {file_name}")
            return None

        text_embeddings = torch.cat([self.get_embeddings(word, f"{file_name}_{i}") for i, word in enumerate(words)], dim=0)
        confidence_scores = torch.tensor(confidences, dtype=torch.float).unsqueeze(1).to(device)
        bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.float).to(device)

        pos_embedding_model = PositionalEmbedding(emb_size=128).to(device)
        pos_embeddings = pos_embedding_model(bounding_boxes)

        node_combiner = NodeFeatureCombiner(text_emb_size=768, pos_emb_size=128).to(device)
        node_features = node_combiner(text_embeddings, pos_embeddings, confidence_scores)

        # Create edges between consecutive words in the same line
        edge_index = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if self._distance(bounding_boxes[i], bounding_boxes[j]) < self.distance_threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)

        # Assign node labels
        target_labels = torch.full((len(words),), label_dict["other"], dtype=torch.long).to(device)  # Initialize all labels to "other"
        for key in labels.keys():
            if key in label_dict:
                label_index = label_dict[key]
                # Assign labels to the first few nodes for demonstration
                if label_index < len(target_labels):
                    target_labels[label_index] = label_index

        data = Data(x=node_features, edge_index=edge_index, y=target_labels)

        return data

    def _distance(self, bbox1, bbox2):
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

def prepare_data_loader(img_folder, ocr_folder, label_folder, emb_folder, batch_size=8, shuffle=True, generate_embeddings=True):
    dataset = OCRDataset(img_folder, ocr_folder, label_folder, emb_folder, generate_embeddings)
    
    def collate_fn(batch):
        batch = [data for data in batch if data is not None]
        if len(batch) == 0:  # If all entries are None, return a dummy batch
            return None
        return DataLoader.collate_fn(batch)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return data_loader

# Function to load the model
def load_model(model_path, in_channels, hidden_channels, out_channels, num_gat_layers, num_gcn_layers, heads):
    model = GCNGAT(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_gat_layers=num_gat_layers, num_gcn_layers=num_gcn_layers, heads=heads).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(model, data_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            if batch is None:
                continue
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            _, pred = out.max(dim=1)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Print classification report
    target_names = ["company", "date", "address", "total", "other"]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

# Save model summary function
def save_model_summary(model, input_size, num_nodes, title, logs_folder):
    # Create a dummy input for model summary
    x = torch.randn((num_nodes, input_size)).to(device)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(device)  # Dummy edge index
    dummy_data = (x, edge_index)
    
    # Generate model summary
    model_summary = summary(model, input_data=dummy_data, verbose=0)
    
    # Save model summary to file
    with open(os.path.join(logs_folder, f"{title}_model_summary.txt"), "w") as f:
        f.write(str(model_summary))

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
    embeddings = graph_data.x[:, :-2].detach().cpu().numpy()  # Exclude position info
    plt.imshow(embeddings, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Token Embeddings")

    plt.tight_layout()
    plt.show()

# Model Training
def train_model(generate_embeddings=True):
    train_loader = prepare_data_loader(img_folder, ocr_folder, label_folder, emb_folder, generate_embeddings=generate_embeddings, batch_size=8)
    test_loader = prepare_data_loader(test_img_folder, test_ocr_folder, test_label_folder, test_emb_folder, shuffle=False, generate_embeddings=False)

    # Update the output dimension to match the number of classes
    num_epochs = 50  # Example: 50 epochs
    num_classes = len(label_dict)  # Adjust out_channels as needed
    hidden_channels = 128  # Define hidden channels
    num_gcn_layers = 3  # Define the number of GCN layers
    num_gat_layers = 3  # Define the number of GAT layers
    heads = 8  # Define the number of heads
    model = GCNGAT(in_channels=897, hidden_channels=hidden_channels, out_channels=num_classes, num_gcn_layers=num_gcn_layers, num_gat_layers=num_gat_layers, heads=heads).to(device)
    
    # Log model summary to text file
    save_model_summary(model, input_size=897, num_nodes=4, title=f"GCNGAT_Model__Detached_{hidden_channels}_{num_epochs}", logs_folder=logs_folder)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Lower learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.CrossEntropyLoss()

    scaler = GradScaler()  # For mixed precision training

    
    model.train()

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=logs_folder)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            if batch is None:
                continue
            batch = batch.to(device)
            optimizer.zero_grad()
            with autocast():  # Mixed precision training
                out = model(batch.x, batch.edge_index)
                loss = criterion(out, batch.y)  # Assuming 'batch.y' contains the target labels
            scaler.scale(loss).backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            # Log the activation of each node (here simplified as logging the output)
            for i, node_output in enumerate(out):
                writer.add_scalar(f'Node Output/Node {i}', node_output.mean().item(), epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
        writer.add_scalar('Loss/train', epoch_loss/len(train_loader), epoch)  # Log loss
        scheduler.step(epoch_loss/len(train_loader))  # Update learning rate

    writer.close()  # Close the TensorBoard writer

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    # Evaluate the model
    evaluate_model(model, test_loader)

def visualize_batch():
    test_loader = prepare_data_loader(test_img_folder, test_ocr_folder, test_label_folder, test_emb_folder, shuffle=False, generate_embeddings=False)
    for batch in test_loader:
        if batch is None:
            continue
        tokens = [word for word in batch.x[:, :-2].cpu().numpy()]
        labels = batch.y.tolist()
        image_path = os.path.join(img_folder, f"{batch.batch[0].item()}.png")
        ocr_text_path = os.path.join(ocr_folder, f"{batch.batch[0].item()}.json")
        visualize_graph_and_image(batch, tokens, labels, image_path, ocr_text_path)
        break  # Visualize only the first batch for brevity

if __name__ == "__main__":
    train_flag = False  # Set to True to train the model
    visualize_flag = True  # Set to True to visualize the graph

    if train_flag:
        train_model(generate_embeddings=True)
    
    if visualize_flag:
        test_loader = prepare_data_loader(test_img_folder, test_ocr_folder, test_label_folder, test_emb_folder, shuffle=False, generate_embeddings=False)
        for batch in test_loader:
            if batch is None:
                continue

            # Extract the correct filename
            filename_idx = batch.batch[0].item()  # Assuming the batch index corresponds to the filename index
            filename = test_loader.dataset.file_names[filename_idx]
            tokens = [word for word in batch.x[:, :-2].detach().cpu().numpy()]
            labels = batch.y.tolist()
            image_path = os.path.join(test_img_folder, f"{filename}.jpg")
            ocr_text_path = os.path.join(test_ocr_folder, f"{filename}.json")
            visualize_graph_and_image(batch, tokens, labels, image_path, ocr_text_path)
            break  # Visualize only the first batch for brevity

    if not train_flag and not visualize_flag:
        test_loader = prepare_data_loader(test_img_folder, test_ocr_folder, test_label_folder, test_emb_folder, shuffle=False, generate_embeddings=False)
        model = load_model(model_save_path, in_channels=897, hidden_channels=128, out_channels=len(label_dict), num_gcn_layers=3, num_gat_layers=3, heads=8)
        evaluate_model(model, test_loader)
