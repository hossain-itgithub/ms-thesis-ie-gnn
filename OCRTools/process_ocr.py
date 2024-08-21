import os
import cv2
import json
import re
import numpy as np
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Configuration
IMAGE_DIR = 'test_input_im'
OCR_DIR = 'test_input_ocr'
OUTPUT_IMAGE_DIR = 'test_output_im'
OUTPUT_TEXT_DIR = 'test_output_box'
USE_MULTIPROCESSING = False

def parse_ocr_file(ocr_file):
    with open(ocr_file, 'r', encoding='utf-8') as file:
        data = file.read()

    lines = data.split('\n')
    extracted_data = []
    for line in lines:
        if "Line:" in line:
            bbox_match = re.search(r"Bounding box (\[.*?\])", line)
            text_match = re.search(r"Line: '(.*?)'", line)
            if bbox_match and text_match:
                bbox_str = bbox_match.group(1).replace("'", '"')
                bbox = json.loads(bbox_str)
                text = text_match.group(1)
                extracted_data.append({
                    'bbox': bbox,
                    'text': text
                })
    return extracted_data

def overlay_text_on_image(image, ocr_data):
    for item in ocr_data:
        bbox = item['bbox']
        text = item['text']
        pts = np.array([[point['x'], point['y']] for point in bbox], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(image, text, (pts[0][0][0], pts[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return image

def process_image_and_ocr(image_file, ocr_file, output_image_dir, output_text_dir):
    image = cv2.imread(image_file)
    ocr_data = parse_ocr_file(ocr_file)
    overlayed_image = overlay_text_on_image(image, ocr_data)

    output_image_path = os.path.join(output_image_dir, os.path.basename(image_file))
    cv2.imwrite(output_image_path, overlayed_image)

    extracted_lines = []
    for item in ocr_data:
        bbox = item['bbox']
        text = item['text']
        extracted_lines.append(f"{bbox[0]['x']},{bbox[0]['y']},{bbox[1]['x']},{bbox[1]['y']},{bbox[2]['x']},{bbox[2]['y']},{bbox[3]['x']},{bbox[3]['y']}, {text}")

    output_text_path = os.path.join(output_text_dir, os.path.basename(ocr_file))
    with open(output_text_path, 'w', encoding='utf-8') as output_file:
        output_file.write("\n".join(extracted_lines))

def process_files_in_directory(image_dir, ocr_dir, output_image_dir, output_text_dir, use_multiprocessing=True):
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_text_dir):
        os.makedirs(output_text_dir)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    ocr_files = [f for f in os.listdir(ocr_dir) if f.endswith('.txt')]

    tasks = []
    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        ocr_file = f"{base_name}.txt"
        if ocr_file in ocr_files:
            tasks.append((os.path.join(image_dir, image_file), os.path.join(ocr_dir, ocr_file), output_image_dir, output_text_dir))

    if use_multiprocessing:
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            list(tqdm(executor.map(lambda p: process_image_and_ocr(*p), tasks), total=len(tasks)))
    else:
        for task in tqdm(tasks):
            process_image_and_ocr(*task)

if __name__ == "__main__":
    process_files_in_directory(IMAGE_DIR, OCR_DIR, OUTPUT_IMAGE_DIR, OUTPUT_TEXT_DIR, USE_MULTIPROCESSING)