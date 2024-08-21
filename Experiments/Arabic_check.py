import os
import json
import shutil
from langdetect import detect, LangDetectException

def check_and_convert_encoding(file_path, output_path, encoding='utf-8'):
    """
    Check if a file is encoded in the specified encoding, and if not, convert it to that encoding.
    Save the converted file to the output path.
    """
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            content = file.read()
        language = detect_language(content)
        
        # If language is not English or Arabic, attempt conversion
        if language not in ['en', 'ar']:
            print(f"{file_path}: Detected language '{language}' not supported. Attempting to re-encode.")
            convert_and_save(file_path, output_path, encoding)
            return
        
        # Write the content to the new output path
        with open(output_path, 'w', encoding=encoding) as file:
            file.write(content)
        print(f"{file_path} -> {output_path}: Encoding is already {encoding}. File copied.")
        
    except (UnicodeDecodeError, LangDetectException):
        # If a UnicodeDecodeError or language detection fails, attempt re-encoding
        print(f"{file_path}: Encoding is not {encoding}. Converting...")
        convert_and_save(file_path, output_path, encoding)

def convert_and_save(file_path, output_path, encoding='utf-8'):
    """
    Attempt to convert file to UTF-8 from different encodings and save.
    """
    with open(file_path, 'rb') as file:
        content = file.read()
    # Try to decode with different encodings and re-encode to UTF-8
    for enc in ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be']:
        try:
            content = content.decode(enc)
            with open(output_path, 'w', encoding=encoding) as file:
                file.write(content)
            print(f"{file_path} -> {output_path}: Successfully converted from {enc} to {encoding}.")
            return
        except (UnicodeDecodeError, LangDetectException):
            continue
    print(f"{file_path}: Failed to convert.")

def detect_language(text):
    """
    Detect the language of the given text.
    """
    try:
        return detect(text)
    except LangDetectException:
        return None

def replicate_directory_structure(src_root, dest_root):
    """
    Replicate the directory structure from the source root to the destination root.
    """
    for dirpath, dirnames, filenames in os.walk(src_root):
        # Compute the corresponding path in the new dataset directory
        dest_dirpath = os.path.join(dest_root, os.path.relpath(dirpath, src_root))
        if not os.path.exists(dest_dirpath):
            os.makedirs(dest_dirpath)
        for filename in filenames:
            src_file_path = os.path.join(dirpath, filename)
            dest_file_path = os.path.join(dest_dirpath, filename)
            yield src_file_path, dest_file_path

def process_dataset(src_root, dest_root):
    """
    Process the dataset, checking and converting encoding, and save to a new location.
    """
    # Replicate the directory structure and process files
    for src_file_path, dest_file_path in replicate_directory_structure(src_root, dest_root):
        if src_file_path.endswith('.txt'):
            if 'entities' in src_file_path:
                validate_entities_file(src_file_path, dest_file_path)
            elif 'box' in src_file_path:
                check_and_convert_encoding(src_file_path, dest_file_path)
        else:
            # Copy other types of files directly (e.g., images)
            shutil.copy(src_file_path, dest_file_path)

def validate_entities_file(file_path, output_path, encoding='utf-8'):
    """
    Validate the JSON structure of an entity file and save it to the output path.
    """
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            data = json.load(file)
            # Optionally, check that all required fields are present
            assert all(key in data for key in ['company', 'date', 'address', 'total'])
        # Write the content to the new output path
        with open(output_path, 'w', encoding=encoding) as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"{file_path} -> {output_path}: JSON structure is valid. File copied.")
    except (json.JSONDecodeError, AssertionError, UnicodeDecodeError, LangDetectException):
        print(f"{file_path}: JSON structure is invalid or encoding is incorrect. Attempting to fix...")
        convert_and_save(file_path, output_path)

# Example usage


# Example usage
src_dataset_dir = 'CoRU_Train'
dest_dataset_dir = 'CoRU_Train_Val'

process_dataset(src_dataset_dir, dest_dataset_dir)
