import os
import time
import concurrent.futures
import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


# # Set the values of your computer vision endpoint and computer vision key
# # as environment variables:
try:
    endpoint = ''
    key = ''
except KeyError:
    print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
    print("Set them before running this sample.")
    exit()

# Set the values of your computer vision endpoint and computer vision key
# as environment variables:
# try:
#     endpoint = 'https://sk-testing-ocr.cognitiveservices.azure.com/'
#     key = '0b8bce726717461b88485fbb7936cfd6'
# except KeyError:
#     print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
#     print("Set them before running this sample.")
#     exit()



# Configurable variables
use_all_cores = True
requests_per_minute = 595
input_directory = "images"
output_directory = "raw_azure_output"
box_coordinates_directory = "box"
num_files_to_process = 100
processed_files_list = "processed_files.txt"

# Create an Image Analysis client for synchronous operations
client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

# Function to process an image file
def process_image_file(image_filename):
    with open(image_filename, "rb") as f:
        image_data = f.read()

    result = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ]
    )

    output_filename = os.path.join(output_directory, os.path.splitext(os.path.basename(image_filename))[0] + ".txt")
    box_coordinates_filename = os.path.join(box_coordinates_directory, os.path.splitext(os.path.basename(image_filename))[0] + ".txt")
    
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(box_coordinates_directory, exist_ok=True)

    with open(output_filename, "w") as output_file, open(box_coordinates_filename, "w") as box_file:
        output_file.write("Image analysis results:\n")
        output_file.write(" Read:\n")
        if result.read is not None:
            for line in result.read.blocks[0].lines:
                output_file.write(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}\n")
                coordinates = ",".join([f"{point.x},{point.y}" for point in line.bounding_polygon])
                box_file.write(f"{coordinates}, {line.text}\n")
                for word in line.words:
                    output_file.write(f"     Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}\n")

# Rate limiting
delay_between_requests = 60.0 / requests_per_minute

# Load processed files
if os.path.exists(processed_files_list):
    with open(processed_files_list, "r") as f:
        processed_files = set(f.read().splitlines())
else:
    processed_files = set()

# Get list of image files in the input directory, filtering out already processed ones
image_files = [os.path.join(input_directory, file) for file in os.listdir(input_directory) if file.lower().endswith(('.png', '.jpg', '.jpeg')) and file not in processed_files]

# Limit the number of files to process
image_files = image_files[:num_files_to_process]

def process_images_in_parallel():
    with concurrent.futures.ThreadPoolExecutor(max_workers=None if use_all_cores else 1) as executor:
        future_to_image = {executor.submit(process_image_file, image_file): image_file for image_file in image_files}
        for future in concurrent.futures.as_completed(future_to_image):
            image_file = future_to_image[future]
            try:
                future.result()
                # Mark the file as processed
                with open(processed_files_list, "a") as f:
                    f.write(f"{os.path.basename(image_file)}\n")
                processed_files.add(os.path.basename(image_file))
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
            time.sleep(delay_between_requests)

if __name__ == "__main__":
    process_images_in_parallel()
