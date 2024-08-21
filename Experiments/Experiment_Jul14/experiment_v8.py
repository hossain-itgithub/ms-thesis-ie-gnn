import os
import json
import re
import shutil
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.nn import GCNConv, GATConv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics import classification_report
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import Counter
from transformers import BertTokenizer, BertModel, XLMRobertaTokenizer, XLMRobertaModel
from torchinfo import summary
from datetime import datetime
import time
from fuzzywuzzy import fuzz

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpatialAttentionLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=1):
        super(SpatialAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"

        self.query = torch.nn.Linear(in_channels, out_channels)
        self.key = torch.nn.Linear(in_channels, out_channels)
        self.value = torch.nn.Linear(in_channels, out_channels)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, edge_index):
        query = self.query(x).view(-1, self.num_heads, self.head_dim)
        key = self.key(x).view(-1, self.num_heads, self.head_dim)
        value = self.value(x).view(-1, self.num_heads, self.head_dim)

        score = torch.einsum("bhq,bhk->bhk", [query, key]) * self.scale
        attention = F.softmax(score, dim=-1)

        out = torch.einsum("bhk,bhv->bhv", [attention, value])
        out = out.contiguous().view(-1, self.num_heads * self.head_dim)

        return out


class ConfigurableGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=64, num_gcn_layers=2, num_gat_layers=2, use_spatial_attention=False, num_gat_heads=1, num_spatial_attention_heads=1):
        super(ConfigurableGNN, self).__init__()
        self.use_spatial_attention = use_spatial_attention

        self.gcn_layers = torch.nn.ModuleList()
        for i in range(num_gcn_layers):
            in_channels = num_node_features if i == 0 else hidden_channels
            self.gcn_layers.append(GCNConv(in_channels, hidden_channels))

        self.gat_layers = torch.nn.ModuleList()
        for i in range(num_gat_layers):
            in_channels = num_node_features if i == 0 and num_gcn_layers == 0 else hidden_channels
            self.gat_layers.append(GATConv(in_channels, hidden_channels, heads=num_gat_heads, concat=False))

        if use_spatial_attention:
            self.spatial_attention = SpatialAttentionLayer(hidden_channels, hidden_channels, num_heads=num_spatial_attention_heads)

        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index)
            x = F.relu(x)

        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index)
            x = F.relu(x)

        if self.use_spatial_attention:
            x = self.spatial_attention(x, edge_index)

        x = self.lin(x)
        return x

def visualize_graph(data, labels):
    G = nx.Graph()
    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        G.add_edge(u, v)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='skyblue')
    
    # Draw labels
    node_labels = {i: labels[i] for i in range(len(labels))}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='red')
    
    plt.show()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()

def train(model, loader, optimizer, criterion, train_batch_sizes):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        total_correct += pred.eq(data.y).sum().item()
        total_samples += data.y.size(0)
        train_batch_sizes.append(data.y.size(0))
    accuracy = total_correct / total_samples
    return total_loss / len(loader), accuracy

def evaluate(model, loader, criterion, eval_batch_sizes):
    model.eval()
    total_loss = 0
    total_correct = 0
    all_preds = []
    all_labels = []
    total_samples = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        total_correct += pred.eq(data.y).sum().item()
        all_preds.append(pred.cpu().numpy())
        all_labels.append(data.y.cpu().numpy())
        total_samples += data.y.size(0)
        eval_batch_sizes.append(data.y.size(0))
    accuracy = total_correct / total_samples
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    report = classification_report(all_labels, all_preds, target_names=['company', 'date', 'address', 'total', 'other'])
    return total_loss / len(loader), accuracy, report

class BaseInvoiceDataset(Dataset):
    def __init__(self, ocr_dir, gt_dir, error_dir, bert_emb_dir, regenerate_bert_embeddings=False, use_xmlroberta=False, transform=None, pre_transform=None):
        super(BaseInvoiceDataset, self).__init__(None, transform, pre_transform)
        self.ocr_dir = ocr_dir
        self.gt_dir = gt_dir
        self.error_dir = error_dir
        self.bert_emb_dir = bert_emb_dir
        self.use_xmlroberta = use_xmlroberta
        
        # Select tokenizer and model based on the flag
        if self.use_xmlroberta:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
            self.embedding_model = XLMRobertaModel.from_pretrained('xlm-roberta-base').to(device)
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.embedding_model = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)
        
        self.error_logs = []
        self.data_list = self.process_data(regenerate_bert_embeddings)
        self.validate_classification_counts()

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def process_data(self, regenerate_bert_embeddings):
        data_list = []
        self.file_classification_mapping = {}
        for file_name in os.listdir(self.ocr_dir):
            if file_name.endswith('.txt'):
                ocr_file_path = os.path.join(self.ocr_dir, file_name)
                gt_file_path = os.path.join(self.gt_dir, file_name.replace('.txt', '.txt'))
                if os.path.exists(gt_file_path):
                    try:
                        with open(ocr_file_path, 'r', encoding='utf-8') as f:
                            ocr_lines = f.readlines()
                    except UnicodeDecodeError:
                        try:
                            with open(ocr_file_path, 'r', encoding='latin1') as f:
                                ocr_lines = f.readlines()
                        except UnicodeDecodeError:
                            with open(ocr_file_path, 'r', encoding='utf-16-le') as f:
                                ocr_lines = f.readlines()
                    
                    try:
                        with open(gt_file_path, 'r', encoding='utf-8') as f:
                            ground_truth = json.load(f)
                    except UnicodeDecodeError:
                        with open(gt_file_path, 'r', encoding='utf-16-le') as f:
                            ground_truth = json.load(f)
                    
                    data = self.process_single_file(ocr_lines, ground_truth, file_name, regenerate_bert_embeddings)
                    if data:
                        data_list.append(data)
                        self.file_classification_mapping[file_name] = data.y.numpy()
        return data_list

    def process_single_file(self, ocr_lines, ground_truth, file_name, regenerate_bert_embeddings):
        node_features = []
        node_texts = []
        
        for line in ocr_lines:
            if line.startswith('----end of file----'):
                break
            parts = line.strip().split(',')
            if len(parts) < 9 or any(part == '' for part in parts[:8]):
                continue  # Skip lines with invalid bounding box data
            try:
                bbox = list(map(int, parts[:8]))
            except ValueError:
                continue  # Skip lines with non-integer bounding box data
            text = ','.join(parts[8:])  # Join the rest of the parts as the text
            node_features.append(bbox)
            node_texts.append(text)

        if not node_features:
            return None

        text_embeddings = self.get_text_embeddings(node_texts, file_name, regenerate_bert_embeddings)
        node_features = np.array([np.concatenate([bbox, emb]) for bbox, emb in zip(node_features, text_embeddings)])

        edges = self.create_edges(node_features)
        labels = [self.get_label(text, ground_truth, file_name) for text in node_texts]

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        y = torch.tensor(labels, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data.file_path = file_name  # Add file path for visualization
        return data

    def create_edges(self, node_features):
        centers = np.array([((bbox[0] + bbox[4]) / 2, (bbox[1] + bbox[5]) / 2) for bbox in node_features])
        dist_matrix = distance_matrix(centers, centers)
        mst = minimum_spanning_tree(dist_matrix).toarray()
        edges = np.vstack(np.nonzero(mst)).T.tolist()
        return edges

    def get_text_embeddings(self, texts, file_name, regenerate_bert_embeddings):
        emb_file = os.path.join(self.bert_emb_dir, file_name.replace('.txt', '_embeddings.npy'))
        if not regenerate_bert_embeddings and os.path.exists(emb_file):
            embeddings = np.load(emb_file)
        else:
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Use the [CLS] token representation
            np.save(emb_file, embeddings)
        return embeddings

    def normalize_text(self, text):
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'[^\w\s]', '', text)  # Remove all non-word characters except space
        return text

    def get_label(self, text, ground_truth, file_name):
        raw_text = text.strip().lower()
        text_cleaned = self.normalize_text(text)
        company_cleaned = self.normalize_text(ground_truth.get('company', ''))
        date_cleaned = self.normalize_text(ground_truth.get('date', ''))
        address_cleaned = self.normalize_text(ground_truth.get('address', ''))
        total_cleaned = self.normalize_text(ground_truth.get('total', ''))

        # Split the address into parts to handle multi-line addresses
        address_parts = [self.normalize_text(part) for part in ground_truth.get('address', '').split(',')]

        if company_cleaned in text_cleaned or fuzz.partial_ratio(company_cleaned, text_cleaned) > 95:
            return 0
        elif date_cleaned in text_cleaned or fuzz.partial_ratio(date_cleaned, text_cleaned) > 95:
            return 1
        elif any(fuzz.partial_ratio(part, text_cleaned) > 95 for part in address_parts) or fuzz.partial_ratio(address_cleaned, text_cleaned) > 95:
            return 2
        elif total_cleaned in text_cleaned or fuzz.partial_ratio(total_cleaned, text_cleaned) > 95:
            return 3
        else:
            error_log = {
                'file_name': file_name,
                'ground_truth': ground_truth,
                'raw_text': raw_text,
                'normalized_text': text_cleaned
            }
            self.error_logs.append(error_log)
            return 4  # Example: dummy label for non-matching text

    def validate_classification_counts(self):
        missing_fields = []
        total_files = len(self.file_classification_mapping)
        used_files = 0

        for file_name, labels in self.file_classification_mapping.items():
            unique_labels = set(labels)
            expected_labels = set(range(4))  # 0: company, 1: date, 2: address, 3: total

            # Load the ground truth data
            ground_truth = self.load_ground_truth(file_name)

            gt_fields = ['company', 'date', 'address', 'total']
            missing_in_gt = [field for field in gt_fields if not ground_truth.get(field)]
            
            if missing_in_gt:
                print(f"Warning: Ground truth missing fields {missing_in_gt} in {file_name}.")
                continue  # Skip validation for this file

            missing = list(expected_labels - unique_labels)

            if missing:
                missing_fields.append({
                    'file_name': file_name,
                    'missing_fields': [class_name for i, class_name in enumerate(['company', 'date', 'address', 'total']) if i in missing]
                })
            else:
                used_files += 1

        # Log missing fields
        if missing_fields:
            with open('missing_fields_log.txt', 'w') as log_file:
                log_file.write(f"Missing fields in files:\n{missing_fields}")
            print(f"Logged missing fields for further analysis.")

        # Summary of files used
        print(f"Used {used_files} out of {total_files} files.")

        # Optionally, return the number of used and total files for further processing or analysis
        return used_files, total_files



    def log_errors(self):
        with open('error_logs.json', 'w') as f:
            json.dump(self.error_logs, f, indent=4)

    def move_to_error_dir(self, file_name):
        os.makedirs(self.error_dir, exist_ok=True)
        ocr_src = os.path.join(self.ocr_dir, file_name)
        ocr_dst = os.path.join(self.error_dir, file_name)
        gt_src = os.path.join(self.gt_dir, file_name.replace('.txt', '.txt'))
        gt_dst = os.path.join(self.error_dir, file_name.replace('.txt', '.txt'))
        shutil.move(ocr_src, ocr_dst)
        shutil.move(gt_src, gt_dst)

    def load_ground_truth(self, file_name):
        gt_file_path = os.path.join(self.gt_dir, file_name)
        try:
            with open(gt_file_path, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
            return ground_truth
        except FileNotFoundError:
            print(f"Ground truth file {file_name} not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {gt_file_path}.")
            return {}

class InvoiceDataset(BaseInvoiceDataset):
    def __init__(self, ocr_dir, gt_dir, error_dir, bert_emb_dir, regenerate_bert_embeddings=False, use_xmlroberta=False, transform=None, pre_transform=None):
        super(InvoiceDataset, self).__init__(ocr_dir, gt_dir, error_dir, bert_emb_dir, regenerate_bert_embeddings, use_xmlroberta, transform, pre_transform)


class TestInvoiceDataset(BaseInvoiceDataset):
    def __init__(self, ocr_dir, gt_dir, error_dir, bert_emb_dir, regenerate_bert_embeddings=False, use_xmlroberta=False, transform=None, pre_transform=None):
        super(TestInvoiceDataset, self).__init__(ocr_dir, gt_dir, error_dir, bert_emb_dir, regenerate_bert_embeddings, use_xmlroberta, transform, pre_transform)

def count_classes(loader):
    all_labels = []
    for data in loader:
        all_labels.append(data.y.numpy())
    all_labels = np.concatenate(all_labels)
    unique, counts = np.unique(all_labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    class_names = ['company', 'date', 'address', 'total', 'other']
    labeled_class_counts = {class_names[key]: value for key, value in class_counts.items()}
    return labeled_class_counts

def main(resume_training=False, resume_config_path=None, resume_model_path=None, use_xmlroberta=False):
    if resume_training:
        with open(resume_config_path, 'r') as f:
            model_config = json.load(f)
    else:
        # Configuration
        model_config = {
            "num_node_features": 8 + 768,  # Example: bbox coordinates + BERT embeddings
            "num_classes": 5,  # Example: company, date, address, total, other
            "hidden_channels": 5000,
            "num_gcn_layers": 3,
            "num_gat_layers": 3,
            "use_spatial_attention": False,
            "num_gat_heads": 4,  # Example number of GAT heads
            "num_spatial_attention_heads": 1,  # Example number of spatial attention heads
            "learning_rate": 0.00008,
            "epochs": 200,
            "batch_size": 4,
            "regenerate_bert_embeddings": False,
            "log_training_reports": True,
            "enable_training": True,
            "enable_evaluation": True,
            "use_xmlroberta": use_xmlroberta  # Add this flag to the model configuration
        }

    # File paths
    ocr_dir = 'Datasets/CoRU/train/box'
    gt_dir = 'Datasets/CoRU/train/entities'
    error_dir = 'Datasets/CoRU/error_files'
    bert_emb_dir = 'Datasets/CoRU/bert_embeddings'
    test_ocr_dir = 'Datasets/CoRU/test/box'
    test_gt_dir = 'Datasets/CoRU/test/entities'

    # Timestamp for logs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'logs/{timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'experiment_log.txt')

    if not resume_training:
        # Save model configuration
        config_file = os.path.join(log_dir, 'model_config.json')
        with open(config_file, 'w') as f:
            json.dump({"Configuration": model_config}, f, indent=4)

    # Create dataset and dataloader
    dataset = InvoiceDataset(ocr_dir, gt_dir, error_dir, bert_emb_dir, model_config["regenerate_bert_embeddings"], model_config["use_xmlroberta"])
    loader = DataLoader(dataset, batch_size=model_config["batch_size"], shuffle=True)

    # Create test dataset and dataloader
    test_dataset = TestInvoiceDataset(test_ocr_dir, test_gt_dir, error_dir, bert_emb_dir, model_config["regenerate_bert_embeddings"], model_config["use_xmlroberta"])
    test_loader = DataLoader(test_dataset, batch_size=model_config["batch_size"], shuffle=False)

    # Initialize model, optimizer, and loss function
    model = ConfigurableGNN(
        num_node_features=model_config["num_node_features"],
        num_classes=model_config["num_classes"],
        hidden_channels=model_config["hidden_channels"],
        num_gcn_layers=model_config["num_gcn_layers"],
        num_gat_layers=model_config["num_gat_layers"],
        use_spatial_attention=model_config["use_spatial_attention"],
        num_gat_heads=model_config["num_gat_heads"],
        num_spatial_attention_heads=model_config["num_spatial_attention_heads"]
    ).to(device)

    if resume_training:
        load_model(model, resume_model_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=model_config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    config_file = os.path.join(log_dir, 'model_config.json')
    with open(config_file, 'w') as f:
        json.dump(model_config, f, indent=4)

    # Count classes
    class_counts = count_classes(loader)
    print(f'Class counts: {class_counts}')
    with open(log_file, 'a') as f:
        f.write(f'Class counts: {class_counts}\n')

    # Model summary
    sample_data = next(iter(loader))
    model_summary = summary(model, input_data=[sample_data.x.to(device), sample_data.edge_index.to(device)], col_names=["input_size", "output_size", "num_params"])
    print(model_summary)
    with open(log_file, 'a') as f:
        f.write(f'Model summary:\n{model_summary}\n')

    # Log initial batch sizes
    train_batch_sizes = []
    eval_batch_sizes = []

    # To store evaluation reports
    eval_reports = []

    # To store training logs
    train_logs = []

    if model_config["enable_training"]:
        # Training loop
        epoch_times = []
        for epoch in range(1, model_config["epochs"] + 1):
            start_time = time.time()
            train_loss, train_accuracy = train(model, loader, optimizer, criterion, train_batch_sizes)
            end_time = time.time()
            epoch_times.append(end_time - start_time)
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_time = (model_config["epochs"] - epoch) * avg_epoch_time / 3600  # Convert seconds to hours
            print(f'Epoch: {epoch}, Loss: {train_loss}, Accuracy: {train_accuracy}, ETA: {remaining_time:.2f} hours')
            train_logs.append(f'Epoch: {epoch}, Loss: {train_loss}, Accuracy: {train_accuracy}, ETA: {remaining_time:.2f} hours\n')
            if model_config["log_training_reports"] and (epoch % 20 == 0 or epoch == model_config["epochs"]):
                eval_loss, accuracy, report = evaluate(model, loader, criterion, eval_batch_sizes)
                print(f'Evaluation Loss: {eval_loss}, Accuracy: {accuracy}')
                print(f'Classification Report:\n{report}')
                eval_reports.append(f'Epoch: {epoch}\nEvaluation Loss: {eval_loss}\nAccuracy: {accuracy}\nClassification Report:\n{report}\n')

        # Dump all training logs at the end of training
        with open(log_file, 'a') as f:
            for log in train_logs:
                f.write(log)

        # Dump all evaluation reports at the end of training
        if model_config["log_training_reports"]:
            with open(log_file, 'a') as f:
                for report in eval_reports:
                    f.write(report)

    if model_config["enable_evaluation"]:
        # Final evaluation
        eval_loss, accuracy, report = evaluate(model, loader, criterion, eval_batch_sizes)
        print(f'Final Evaluation Loss: {eval_loss}, Accuracy: {accuracy}')
        print(f'Final Classification Report:\n{report}')
        with open(log_file, 'a') as f:
            f.write(f'Final Evaluation Loss: {eval_loss}, Accuracy: {accuracy}\n')
            f.write(f'Final Classification Report:\n{report}\n')

        # Test evaluation
        test_loss, test_accuracy, test_report = evaluate(model, test_loader, criterion, eval_batch_sizes)
        print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
        print(f'Test Classification Report:\n{test_report}')
        with open(log_file, 'a') as f:
            f.write(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}\n')
            f.write(f'Test Classification Report:\n{test_report}\n')

    # Visualize graph
    visualize_graph(sample_data, sample_data.y.numpy())

    # Save model
    model_path = os.path.join(log_dir, 'invoice_gnn.pth')
    save_model(model, model_path)
    with open(log_file, 'a') as f:
        f.write(f'Model saved to {model_path}\n')

    # Save model parameters
    model_params_path = os.path.join(log_dir, 'model_params.txt')
    with open(model_params_path, 'w') as f:
        f.write(str(model))
    with open(log_file, 'a') as f:
        f.write(f'Model parameters saved to {model_params_path}\n')

    # Log errors
    dataset.log_errors()
    test_dataset.log_errors()
    with open(log_file, 'a') as f:
        f.write('Errors logged.\n')

    # Report percentage of erroneous files
    total_files = len(os.listdir(ocr_dir)) + len(os.listdir(test_ocr_dir)) + len(os.listdir(error_dir))
    error_files = len(os.listdir(error_dir))
    error_percentage = (error_files / total_files) * 100
    print(f'Percentage of erroneous files: {error_percentage:.2f}%')
    with open(log_file, 'a') as f:
        f.write(f'Percentage of erroneous files: {error_percentage:.2f}%\n')

if __name__ == "__main__":
    resume_training = False
    resume_config_path = 'logs/20240802_175438/model_config.json'
    resume_model_path = 'logs/20240802_175438/invoice_gnn.pth'

    # You can set `use_xmlroberta=True` to switch to XMLRoBERTa embeddings
    main(resume_training=resume_training, resume_config_path=resume_config_path, resume_model_path=resume_model_path, use_xmlroberta=False)
