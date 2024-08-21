import os
import json
import re
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ExifTags, ImageOps
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from collections import defaultdict
from torch_geometric.loader import DataLoader
from my_model2 import ConfigurableGNN, InvoiceDataset, device  # Adjust imports based on your file structure

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        if re.match(r'Epoch: \d+, Loss:', line):
            parts = line.split(',')
            epoch = int(parts[0].split(':')[1].strip())
            train_loss = float(parts[1].split(':')[1].strip())
            train_accuracy = float(parts[2].split(':')[1].strip())
            epochs.append(epoch)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
        elif re.match(r'Epoch: \d+\nEvaluation Loss:', line):
            eval_epoch_line = line
            eval_loss_line = lines[lines.index(line) + 1]
            eval_report_lines = lines[lines.index(line) + 2:lines.index(line) + 8]  # Adjust based on actual length of report
            
            eval_epoch = int(eval_epoch_line.split()[1])
            eval_loss = float(eval_loss_line.split()[2])
            eval_accuracy = float(eval_loss_line.split()[4])
            
            if eval_epoch not in epochs:
                epochs.append(eval_epoch)
            eval_losses.append(eval_loss)
            eval_accuracies.append(eval_accuracy)

    # Ensure all lists have the same length by padding with None or appropriate values
    max_length = max(len(epochs), len(train_losses), len(train_accuracies), len(eval_losses), len(eval_accuracies))
    while len(train_losses) < max_length:
        train_losses.append(None)
    while len(train_accuracies) < max_length:
        train_accuracies.append(None)
    while len(eval_losses) < max_length:
        eval_losses.append(None)
    while len(eval_accuracies) < max_length:
        eval_accuracies.append(None)

    logging.debug(f"Epochs: {epochs}")
    logging.debug(f"Train Losses: {train_losses}")
    logging.debug(f"Train Accuracies: {train_accuracies}")
    logging.debug(f"Eval Losses: {eval_losses}")
    logging.debug(f"Eval Accuracies: {eval_accuracies}")

    return epochs, train_losses, train_accuracies, eval_losses, eval_accuracies

def load_model_config(config_file):
    logging.info(f"Loading model configuration from: {config_file}")
    with open(config_file, 'r') as f:
        config = json.load(f)["Configuration"]
    return config

def plot_metrics(all_loss_metrics, all_accuracy_metrics, log_dir):
    plt.figure()
    for metrics in all_loss_metrics:
        epochs, train_losses, eval_losses, label = metrics
        # Filter out None values
        filtered_epochs = [e for e, l in zip(epochs, train_losses) if l is not None]
        filtered_train_losses = [l for l in train_losses if l is not None]
        filtered_eval_losses = [l for l in eval_losses if l is not None]
        
        if filtered_epochs and filtered_train_losses:
            plt.plot(filtered_epochs, filtered_train_losses, label=f'Train Loss ({label})', marker='o')
        if filtered_epochs and filtered_eval_losses:
            plt.plot(filtered_epochs, filtered_eval_losses, label=f'Eval Loss ({label})', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(log_dir, 'loss_plot.pdf'), format="pdf", bbox_inches='tight')
    plt.close()

    plt.figure()
    for metrics in all_accuracy_metrics:
        epochs, train_accuracies, eval_accuracies, label = metrics
        # Filter out None values
        filtered_epochs = [e for e, a in zip(epochs, train_accuracies) if a is not None]
        filtered_train_accuracies = [a for a in train_accuracies if a is not None]
        filtered_eval_accuracies = [a for a in eval_accuracies if a is not None]
        
        if filtered_epochs and filtered_train_accuracies:
            plt.plot(filtered_epochs, filtered_train_accuracies, label=f'Train Accuracy ({label})', marker='o')
        if filtered_epochs and filtered_eval_accuracies:
            plt.plot(filtered_epochs, filtered_eval_accuracies, label=f'Eval Accuracy ({label})', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(log_dir, 'accuracy_plot.pdf'), format="pdf", bbox_inches='tight')
    plt.close()

def find_log_files(directory):
    log_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('experiment_log.txt'):
                log_files.append(os.path.join(root, file))
    return log_files

def correct_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
            image = ImageOps.exif_transpose(image)
    except (AttributeError, KeyError, IndexError):
        pass
    return image

def adjust_bounding_boxes_for_orientation(boxes, orientation, image_size):
    width, height = image_size
    adjusted_boxes = []
    for box in boxes:
        if orientation == 3:
            adjusted_box = [width - box[4], height - box[5], width - box[0], height - box[1]]
        elif orientation == 6:
            adjusted_box = [height - box[5], box[0], height - box[1], box[4]]
        elif orientation == 8:
            adjusted_box = [box[1], width - box[4], box[5], width - box[0]]
        else:
            adjusted_box = box
        adjusted_boxes.append(adjusted_box)
    return adjusted_boxes

def draw_bounding_boxes(image, boxes, labels, colors, class_names):
    # Get orientation from EXIF data
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation)
        else:
            orientation = 1
    except (AttributeError, KeyError, IndexError):
        orientation = 1

    # Adjust bounding boxes based on orientation
    boxes = adjust_bounding_boxes_for_orientation(boxes, orientation, image.size)

    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        color = colors[label]
        # Ensure box coordinates are in correct format
        x0, y0 = min(box[0], box[2]), min(box[1], box[3])
        x1, y1 = max(box[0], box[2]), max(box[1], box[3])
        box_coords = [x0, y0, x1, y1]
        #logging.debug(f'Drawing box: {box_coords} with label: {class_names[label]}')
        draw.rectangle(box_coords, outline=color, width=2)
        draw.text((x0, y0), class_names[label], fill=color)
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
    plt.savefig(os.path.join(log_dir, f'confusion_matrix_{label}.pdf'), format="pdf")
    plt.close()

def main():
    log_dir = 'Analysis/Logs_Test'  # Replace with the actual directory containing log files
    error_dir = 'Analysis/Errors'  # Replace with the actual error directory

    ocr_dir = 'data3/test/box'  # Replace with the actual OCR directory
    gt_dir = 'data3/test/entities'  # Replace with the actual ground truth directory
    bert_emb_dir = 'data3/bert_embeddings'  # Replace with the actual BERT embeddings directory
    img_dir = 'data3/test/img'  # Replace with the actual image directory
    
    log_files = find_log_files(log_dir)
    all_loss_metrics = []
    all_accuracy_metrics = []
    results_per_model = {}

    if not log_files:
        logging.error("No log files found.")
        return

    for log_file in log_files:
        logging.info(f"Processing log file: {log_file}")
        experiment_dir = os.path.dirname(log_file)  # Directory of the current experiment
        config_file = os.path.join(experiment_dir, 'model_config.json')
        config = load_model_config(config_file)
        
        learning_rate = config.get("learning_rate", "unknown")

        label = f'GCN: {config["num_gcn_layers"]}, GAT: {config["num_gat_layers"]}, SA: {config["use_spatial_attention"]}, HC: {config["hidden_channels"]}, LR: {learning_rate}'
        epochs, train_losses, train_accuracies, eval_losses, eval_accuracies = parse_log_file(log_file)
        all_loss_metrics.append((epochs, train_losses, eval_losses, label))
        all_accuracy_metrics.append((epochs, train_accuracies, eval_accuracies, label))
        
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
            use_spatial_attention=config["use_spatial_attention"],
            num_attention_heads=config.get("num_attention_heads", 1)  # Include num_attention_heads
        ).to(device)
        
        model_path = os.path.join(experiment_dir, 'invoice_gnn.pth')
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

        # Ensure experiment directory exists for predictions
        predictions_dir = os.path.join(experiment_dir, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)

        # Plot confusion matrix and classification report
        plot_confusion_matrix(y_true, y_pred, experiment_dir, label)
        report = classification_report(y_true, y_pred, target_names=['company', 'date', 'address', 'total', 'other'])
        logging.info(f"Classification Report for {label}:\n{report}")

        # # Plot annotations comparisons for collected sample data
        for sample_data, _ in sample_data_for_plot:
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            img_path = os.path.join(img_dir, sample_data.file_path[0].replace('.txt', '.jpg'))  # Ensure this gets the first element if it's a list
            img = Image.open(img_path)
            bbox_coordinates = sample_data.x[:, :8].cpu().numpy().tolist()
            class_names = ['company', 'date', 'address', 'total', 'other']
            colors = ['red', 'blue', 'green', 'purple', 'orange']
            
            img = correct_image_orientation(img)
            img_with_gt = draw_bounding_boxes(img.copy(), bbox_coordinates, sample_data.y.cpu().numpy(), colors, class_names)
            axs[0].imshow(img_with_gt)
            axs[0].set_title('Original Image with Ground Truth')
            axs[0].axis('off')
            
            visualize_annotations(sample_data, sample_data.y.cpu().numpy(), 'Graph with Ground Truth', axs[1], colors, class_names)

            predictions = run_inference(model, sample_data)
            img_with_pred = draw_bounding_boxes(img.copy(), bbox_coordinates, predictions, colors, class_names)
            axs[2].imshow(img_with_pred)
            axs[2].set_title('Original Image with Predictions')
            axs[2].axis('off')

            plt.savefig(os.path.join(predictions_dir, f'{sample_data.file_path[0]}.pdf'),format='pdf')
            plt.close('all')

    # Plot metrics for all experiments
    plot_metrics(all_loss_metrics, all_accuracy_metrics, log_dir)

if __name__ == "__main__":
    main()
