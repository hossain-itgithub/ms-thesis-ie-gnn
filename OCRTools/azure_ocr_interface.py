import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# Set the values of your computer vision endpoint and computer vision key
# as environment variables:
try:
    endpoint = ''
    key = ''
except KeyError:
    print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
    print("Set them before running this sample.")
    exit()



# Create an Image Analysis client for synchronous operations
client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

# Load image to analyze into a 'bytes' object
image_filename = "Sample3.jpg"
with open(image_filename, "rb") as f:
    image_data = f.read()

# Extract text (OCR) from an image stream. This will be a synchronously (blocking) call.
result = client.analyze(
    image_data=image_data,
    visual_features=[VisualFeatures.READ]
)

# Prepare the output filename
output_filename = os.path.splitext(image_filename)[0] + ".txt"

# Write text (OCR) analysis results to a file
with open(output_filename, "w") as output_file:
    output_file.write("Image analysis results:\n")
    output_file.write(" Read:\n")
    if result.read is not None:
        for line in result.read.blocks[0].lines:
            output_file.write(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}\n")
            for word in line.words:
                output_file.write(f"     Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}\n")
