import os
import json
import re  # Add this import statement
import shutil
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, GATConv
from transformers import BertTokenizer, BertModel
import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpatialAttentionLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialAttentionLayer, self).__init__()
        self.query = torch.nn.Linear(in_channels, out_channels)
        self.key = torch.nn.Linear(in_channels, out_channels)
        self.value = torch.nn.Linear(in_channels, out_channels)
        self.scale = out_channels ** -0.5

    def forward(self, x, edge_index):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        score = torch.mm(query, key.t()) * self.scale
        attention = F.softmax(score, dim=-1)

        return torch.mm(attention, value)

class ConfigurableGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=64, num_gcn_layers=2, num_gat_layers=2, use_spatial_attention=False):
        super(ConfigurableGNN, self).__init__()
        self.use_spatial_attention = use_spatial_attention

        self.gcn_layers = torch.nn.ModuleList()
        for i in range(num_gcn_layers):
            in_channels = num_node_features if i == 0 else hidden_channels
            self.gcn_layers.append(GCNConv(in_channels, hidden_channels))

        self.gat_layers = torch.nn.ModuleList()
        for i in range(num_gat_layers):
            in_channels = num_node_features if i == 0 and num_gcn_layers == 0 else hidden_channels
            self.gat_layers.append(GATConv(in_channels, hidden_channels))

        if use_spatial_attention:
            self.spatial_attention = SpatialAttentionLayer(hidden_channels, hidden_channels)

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

class BaseInvoiceDataset(Dataset):
    def __init__(self, ocr_dir, gt_dir, error_dir, bert_emb_dir, regenerate_bert_embeddings=False, transform=None, pre_transform=None):
        super(BaseInvoiceDataset, self).__init__(None, transform, pre_transform)
        self.ocr_dir = ocr_dir
        self.gt_dir = gt_dir
        self.error_dir = error_dir
        self.bert_emb_dir = bert_emb_dir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
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
                        with open(ocr_file_path, 'r', encoding='latin1') as f:
                            ocr_lines = f.readlines()
                    
                    with open(gt_file_path, 'r') as f:
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
        node_features = [np.concatenate([bbox, emb]) for bbox, emb in zip(node_features, text_embeddings)]

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
                outputs = self.bert_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Use the [CLS] token representation
            np.save(emb_file, embeddings)
        return embeddings

    def normalize_text(self, text):
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove all non-alphanumeric characters except space
        #text = re.sub(r'[\s-]+', ' ', text)  # Normalize spaces, commas, and dashes to a single space
        text = re.sub(r'[\s.,-]+', ' ', text)  # Normalize spaces, commas, and dashes to a single space
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

        if company_cleaned in text_cleaned:
            return 0
        elif date_cleaned in text_cleaned:
            return 1
        elif any(part in text_cleaned for part in address_parts) or address_cleaned in text_cleaned:
            return 2
        elif total_cleaned in text_cleaned:
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
        for file_name, labels in self.file_classification_mapping.items():
            unique_labels = set(labels)
            if len(unique_labels) < 5:  # Assuming there should be at least one of each label in a well-formed file
                missing = [class_name for i, class_name in enumerate(['company', 'date', 'address', 'total', 'other']) if i not in unique_labels]
                missing_fields.append({
                    'file_name': file_name,
                    'missing_fields': missing
                })
                self.move_to_error_dir(file_name)
        
        if missing_fields:
            raise ValueError(f"Missing fields in files:\n{missing_fields}")

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

class InvoiceDataset(BaseInvoiceDataset):
    def __init__(self, ocr_dir, gt_dir, error_dir, bert_emb_dir, regenerate_bert_embeddings=False, transform=None, pre_transform=None):
        super(InvoiceDataset, self).__init__(ocr_dir, gt_dir, error_dir, bert_emb_dir, regenerate_bert_embeddings, transform, pre_transform)

class TestInvoiceDataset(BaseInvoiceDataset):
    def __init__(self, ocr_dir, gt_dir, error_dir, bert_emb_dir, regenerate_bert_embeddings=False, transform=None, pre_transform=None):
        super(TestInvoiceDataset, self).__init__(ocr_dir, gt_dir, error_dir, bert_emb_dir, regenerate_bert_embeddings, transform, pre_transform)
