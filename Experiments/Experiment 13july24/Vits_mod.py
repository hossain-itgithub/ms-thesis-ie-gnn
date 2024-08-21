
import os
import json
import cv2
import numpy as np
import torch
from transformers import ViTFeatureExtractor, ViTModel
import dgl
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import precision_recall_fscore_support

import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Make sure to download necessary resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

base_path = '/home/riya/MS-Thesis/data4'
train_image_path = os.path.join(base_path, 'train', 'img')
train_bbox_path = os.path.join(base_path, 'train', 'box')
train_entity_path = os.path.join(base_path, 'train', 'entities')

test_image_path = os.path.join(base_path, 'test', 'img')
test_bbox_path = os.path.join(base_path, 'test', 'box')
test_entity_path = os.path.join(base_path, 'test', 'entities')

def normalize_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove accents and normalize unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Tokenize text (keep special characters)
    words = re.findall(r'\b\w+\b|[^\w\s]', text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join words back to a single string
    normalized_text = ' '.join(words)
    
    return normalized_text

def load_annotations(annotation_path):
    annotations = []
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r', encoding='ISO-8859-1') as f:
            for line in f.readlines():
                parts = line.strip().split(',')
                if len(parts) < 9:
                    print(f"Invalid line in {annotation_path}: {line}")
                    continue
                text = ','.join(parts[8:])
                text = normalize_text(text)  # Normalize the text here
                try:
                    bbox = list(map(int, parts[:8]))
                    annotations.append({'text': text, 'bbox': bbox})
                except ValueError as e:
                    print(f"Error parsing bbox in {annotation_path}: {e}")
    else:
        print(f"File not found: {annotation_path}")
    return annotations

def load_text_labels(text_path):
    if os.path.exists(text_path):
        with open(text_path, 'r', encoding='ISO-8859-1') as f:
            labels = f.read().strip().split('\n')
            labels = [normalize_text(label) for label in labels]  # Normalize the labels here
            return labels
    else:
        print(f"File not found: {text_path}")
        return []

def load_dataset(image_path, bbox_path, entity_path):
    images = []
    annotations = []
    labels = []
    for filename in os.listdir(image_path):
        if filename.endswith('.jpg'):
            image_file = os.path.join(image_path, filename)
            bbox_file = os.path.join(bbox_path, filename.replace('.jpg', '.txt'))
            text_file = os.path.join(entity_path, filename.replace('.jpg', '.txt'))
            
            if os.path.exists(image_file) and os.path.exists(bbox_file) and os.path.exists(text_file):
                images.append(cv2.imread(image_file))
                annotations.append(load_annotations(bbox_file))
                labels.append(load_text_labels(text_file))
            else:
                print(f"Missing file(s) for {filename}")
    return images, annotations, labels


train_images, train_annotations, train_labels = load_dataset(train_image_path, train_bbox_path, train_entity_path)
test_images, test_annotations, test_labels = load_dataset(test_image_path, test_bbox_path, test_entity_path)


feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').cuda()


def get_vit_features(images):
    inputs = feature_extractor(images=images, return_tensors="pt").to('cuda')
    outputs = vit_model(**inputs)
    return outputs.last_hidden_state

train_features = [get_vit_features([image]) for image in train_images]
test_features = [get_vit_features([image]) for image in test_images]

def construct_graph(ocr_data, features):
    nodes = []
    edges_from = []
    edges_to = []
    node_features = []
    for idx, data in enumerate(ocr_data):
        nodes.append(data['text'])
        node_features.append(features[0][idx].cpu().detach().numpy())
        for other_idx, other_data in enumerate(ocr_data):
            if idx != other_idx:
                if np.linalg.norm(np.array(data['bbox']) - np.array(other_data['bbox'])) < 50:
                    edges_from.append(idx)
                    edges_to.append(other_idx)
    g = dgl.graph((edges_from, edges_to))
    g.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
    return g

train_graphs = [construct_graph(annotation, feature) for annotation, feature in zip(train_annotations, train_features)]
test_graphs = [construct_graph(annotation, feature) for annotation, feature in zip(test_annotations, test_features)]

#GNN model
class GNN(torch.nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h

gnn_model = GNN(in_feats=768, h_feats=256).cuda()
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

#training loop
for epoch in range(20):
    for graph, label in zip(train_graphs, train_labels):
        features = graph.ndata['feat'].cuda()
        labels = torch.tensor([label[node['text']] for node in graph.ndata], dtype=torch.long).cuda()
        preds = gnn_model(graph.to('cuda'), features)
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()  # Clear GPU memory

# Self-Supervised Learning
def masked_node_prediction(graph, model):
    mask = torch.rand(graph.num_nodes()) > 0.8
    masked_features = graph.ndata['feat'][mask].cuda()
    preds = model(graph.to('cuda'), masked_features)
    return loss_fn(preds, torch.zeros(mask.sum().item(), dtype=torch.long).cuda())

def edge_prediction(graph, model):
    mask = torch.rand(graph.num_edges()) > 0.8
    masked_edges = graph.edges()[mask]
    preds = model(graph.to('cuda'), graph.ndata['feat'].cuda())
    return loss_fn(preds, torch.zeros(mask.sum().item(), dtype=torch.long).cuda())

# pre-training loop
for epoch in range(20):
    for graph in train_graphs:
        loss = masked_node_prediction(graph, gnn_model) + edge_prediction(graph, gnn_model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for epoch in range(20):
    for graph, label in zip(train_graphs, train_labels):
        features = graph.ndata['feat'].cuda()
        labels = torch.tensor([label[node['text']] for node in graph.ndata], dtype=torch.long).cuda()
        preds = gnn_model(graph.to('cuda'), features)
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

all_preds = []
all_labels = []
for graph, label in zip(test_graphs, test_labels):
    features = graph.ndata['feat'].cuda()
    labels = torch.tensor([label[node['text']] for node in graph.ndata], dtype=torch.long).cuda()
    preds = gnn_model(graph.to('cuda'), features)
    all_preds.append(preds.cpu().detach().numpy())
    all_labels.append(labels.cpu().detach().numpy())

precision, recall, f1, _ = precision_recall_fscore_support(np.concatenate(all_labels), np.concatenate(all_preds).argmax(axis=1), average='weighted')
print(f'Precision: {precision}, Recall: {recall}, F1-score: {f1}')
