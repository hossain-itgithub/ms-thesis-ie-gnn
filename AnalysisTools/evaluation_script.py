import os
import json
import re
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from collections import defaultdict
from torch_geometric.loader import DataLoader
from my_model import ConfigurableGNN, InvoiceDataset, device  # Adjust imports based on your file structure

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_log_file(log_file):
    logging.info(f"Parsing log file: {log_file}")
    epochs = []
    train_losses = []
    train_accuracies = []
    eval_losses = []
    eval_accuracies = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith('Epoch:'):
            parts = line.split(',')
            epoch = int(parts[0].split(':')[1].strip())
            train_loss = float(parts[1].split(':')[1].strip())
            train_accuracy = float(parts[2].split(':')[1].strip())
            epochs.append(epoch)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
        elif line.startswith('Evaluation Loss:'):
            eval_loss = float(line.split(',')[0].split(':')[1].strip())
            eval_accuracy = float(line.split(',')[1].split(':')[1].strip())
            eval_losses.append(eval_loss)
            eval_accuracies.append(eval_accuracy)
    
    return epochs, train_losses, train_accuracies, eval_losses, eval_accuracies

def load_model_config(config_file):
    logging.info(f"Loading model configuration from: {config_file}")
    with open(config_file, 'r') as f:
        config = json.load(f)["Configuration"]
    return config

def plot_metrics(all_metrics, log_dir):
    plt.figure()
    for metrics in all_metrics:
        epochs, train_losses, eval_losses, label = metrics
        plt.plot(epochs, train_losses, label=f'Train Loss ({label})', marker='o')
        plt.plot(epochs, eval_losses, label=f'Eval Loss ({label})', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'loss_plot.png'))
    plt.show()

    plt.figure()
    for metrics in all_metrics:
        epochs, train_accuracies, eval_accuracies, label = metrics
        plt.plot(epochs, train_accuracies, label=f'Train Accuracy ({label})', marker='o')
        plt.plot(epochs, eval_accuracies, label=f'Eval Accuracy ({label})', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'accuracy_plot.png'))
    plt.show()

def find_log_files(directory):
    log_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('experiment_log.txt'):
                log_files.append(os.path.join(root, file))
    return log_files

def draw_bounding_boxes(image, boxes, labels, colors, class_names):
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        color = colors[label]
        box_coords = [box[0], box[1], box[4], box[5]]  # Convert from 8-coord to 2-coord format
        draw.rectangle(box_coords, outline=color, width=2)
        draw.text((box[0], box[1]), class_names[label], fill=color)
    return image

def visualize_annotations(data, labels, title, ax, colors, class_names):
    import networkx as nx

    G = nx.Graph()
    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        G.add_edge(u, v)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='skyblue', ax=ax)
    
    node_labels = {i: class_names[labels[i]] for i in range(len(labels))}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='red', ax=ax)
    
    ax.set_title(title)

def run_inference(model, sample_data):
    model.eval()
    sample_data = sample_data.to(device)
    with torch.no_grad():
        output = model(sample_data.x, sample_data.edge_index)
    
    predictions = output.argmax(dim=1).cpu().numpy()
    return predictions

def plot_confusion_matrix(y_true, y_pred, log_dir, label):
    class_names = ['company', 'date', 'address', 'total', 'other']
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix ({label})')
    plt.savefig(os.path.join(log_dir, f'confusion_matrix_{label}.png'))
    plt.show()

def main():
    log_dir = 'Analysis/Logs_Test'  # Replace with the actual directory containing log files
    ocr_dir = 'data3/test/box'  # Replace with the actual OCR directory
    gt_dir = 'data3/test/entities'  # Replace with the actual ground truth directory
    error_dir = 'Analysis/Errors'  # Replace with the actual error directory
    bert_emb_dir = 'data3/bert_embeddings'  # Replace with the actual BERT embeddings directory
    img_dir = 'data3/test/img'  # Replace with the actual image directory
    
    log_files = find_log_files(log_dir)
    all_metrics = []
    results_per_model = {}

    if not log_files:
        logging.error("No log files found.")
        return

    for log_file in log_files:
        logging.info(f"Processing log file: {log_file}")
        config_file = os.path.join(os.path.dirname(log_file), 'model_config.json')
        config = load_model_config(config_file)
        
        label = f'GCN: {config["num_gcn_layers"]}, GAT: {config["num_gat_layers"]}, SA: {config["use_spatial_attention"]}, HC: {config["hidden_channels"]}'
        epochs, train_losses, train_accuracies, eval_losses, eval_accuracies = parse_log_file(log_file)
        all_metrics.append((epochs, train_losses, eval_losses, label))
        all_metrics.append((epochs, train_accuracies, eval_accuracies, label))
        
        y_true = []
        y_pred = []
        sample_data_for_plot = []

        # Load example data from the dataset
        dataset = InvoiceDataset(ocr_dir=ocr_dir, gt_dir=gt_dir, error_dir=error_dir, bert_emb_dir=bert_emb_dir, regenerate_bert_embeddings=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        model = ConfigurableGNN(
            num_node_features=config["num_node_features"],
            num_classes=config["num_classes"],
            hidden_channels=config["hidden_channels"],
            num_gcn_layers=config["num_gcn_layers"],
            num_gat_layers=config["num_gat_layers"],
            use_spatial_attention=config["use_spatial_attention"]
        ).to(device)
        
        model_path = os.path.join(os.path.dirname(log_file), 'invoice_gnn.pth')
        logging.info(f"Loading model from: {model_path}")
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            continue

        for sample_data in loader:
            sample_data_for_plot.append((sample_data, log_file))  # Save sample data and log_file for later plotting
            predictions = run_inference(model, sample_data)
            y_true.extend(sample_data.y.cpu().numpy())
            y_pred.extend(predictions)

        results_per_model[label] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'sample_data_for_plot': sample_data_for_plot
        }

    # Plot confusion matrices and classification reports for each model
    for label, results in results_per_model.items():
        y_true = results['y_true']
        y_pred = results['y_pred']
        plot_confusion_matrix(y_true, y_pred, log_dir, label)
        report = classification_report(y_true, y_pred, target_names=['company', 'date', 'address', 'total', 'other'])
        logging.info(f"Classification Report for {label}:\n{report}")

        # Plot annotations comparisons for collected sample data
        for sample_data, log_file in results['sample_data_for_plot']:
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            img_path = os.path.join(img_dir, sample_data.file_path[0].replace('.txt','.jpg'))  # Ensure this gets the first element if it's a list
            img = Image.open(img_path)
            bbox_coordinates = sample_data.x[:, :8].numpy().tolist()
            class_names = ['company', 'date', 'address', 'total', 'other']
            colors = ['red', 'blue', 'green', 'purple', 'orange']
            
            img_with_gt = draw_bounding_boxes(img.copy(), bbox_coordinates, sample_data.y.numpy(), colors, class_names)
            axs[0].imshow(img_with_gt)
            axs[0].set_title('Original Image with Ground Truth')
            axs[0].axis('off')
            
            visualize_annotations(sample_data, sample_data.y.numpy(), 'Graph with Ground Truth', axs[1], colors, class_names)

            # Reload model for predictions
            config_file = os.path.join(os.path.dirname(log_file), 'model_config.json')
            config = load_model_config(config_file)
            model = ConfigurableGNN(
                num_node_features=config["num_node_features"],
                num_classes=config["num_classes"],
                hidden_channels=config["hidden_channels"],
                num_gcn_layers=config["num_gcn_layers"],
                num_gat_layers=config["num_gat_layers"],
                use_spatial_attention=config["use_spatial_attention"]
            ).to(device)
            
            model.load_state_dict(torch.load(model_path, map_location=device))
            predictions = run_inference(model, sample_data)

            img_with_pred = draw_bounding_boxes(img.copy(), bbox_coordinates, predictions, colors, class_names)
            axs[2].imshow(img_with_pred)
            axs[2].set_title('Original Image with Predictions')
            axs[2].axis('off')

            plt.savefig(os.path.join(os.path.dirname(log_file), f'annotations_comparison_{label}.png'))
            plt.show()

    # Plot metrics
    plot_metrics(all_metrics, log_dir)

if __name__ == "__main__":
    main()
