import chainlit as cl
import cv2
import numpy as np
import json
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use relative paths
BASE_DIR = Path(__file__).parent
PLANOGRAM_JSON = BASE_DIR / 'data' / 'product_positions_adjusted_v10.json'
PLANOGRAM_IMAGE = BASE_DIR / 'data' / 'shelf_overlay_adjusted_v10.jpg'

def validate_image(image, name="image"):
    """Validate that an image is properly loaded and has valid dimensions."""
    if image is None:
        raise ValueError(f"Failed to load {name}")
    if image.size == 0:
        raise ValueError(f"{name} is empty")
    if len(image.shape) != 3:
        raise ValueError(f"{name} must be a color image (3 channels)")
    return True

# Load planogram metadata
try:
    with open(PLANOGRAM_JSON, 'r') as f:
        planogram_data = json.load(f)
    if not planogram_data:
        raise ValueError("Planogram data is empty")
    logger.info(f"Successfully loaded planogram data from {PLANOGRAM_JSON}")
except FileNotFoundError:
    logger.error(f"Planogram JSON file not found at {PLANOGRAM_JSON}")
    raise
except json.JSONDecodeError:
    logger.error(f"Invalid JSON in planogram file {PLANOGRAM_JSON}")
    raise

# Load planogram image
try:
    planogram_image = cv2.imread(str(PLANOGRAM_IMAGE))
    validate_image(planogram_image, "planogram image")
    logger.info(f"Successfully loaded planogram image from {PLANOGRAM_IMAGE}")
except Exception as e:
    logger.error(f"Error loading planogram image: {str(e)}")
    raise

@cl.on_chat_start
async def start():
    await cl.Message(
        "📸 Welcome to the Shelf Checker App!\n\n"
        "Upload your shelf photo below and I'll check for any missing products.\n"
        "Make sure the photo is well-lit and shows the entire shelf clearly."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    try:
        # Check if message has files
        if not message.elements:
            await cl.Message("Please upload a shelf photo to analyze.").send()
            return

        # Get the first file
        file = message.elements[0]
        if not file.path.endswith(('.jpg', '.jpeg', '.png')):
            await cl.Message("Please upload a JPG or PNG image file.").send()
            return

        # Show processing message
        await cl.Message("⏳ Processing your image...").send()
        
        # Read and validate image
        shelf_img = cv2.imread(file.path)
        validate_image(shelf_img, "uploaded image")
        
        # Check if image dimensions match planogram
        h_plan, w_plan = planogram_image.shape[:2]
        h_shelf, w_shelf = shelf_img.shape[:2]
        
        if abs(h_plan - h_shelf) > 50 or abs(w_plan - w_shelf) > 50:
            logger.warning(f"Image dimensions mismatch: Planogram {w_plan}x{h_plan}, Shelf {w_shelf}x{h_shelf}")
            await cl.Message(
                "⚠️ Warning: The uploaded image dimensions don't match the planogram.\n"
                "This might affect the accuracy of the results."
            ).send()
        
        # Process image and get report
        report = check_empty_spaces(shelf_img)
        
        # Send results as a new message
        await cl.Message(report).send()
        
    except Exception as e:
        logger.error(f"Error processing uploaded image: {str(e)}")
        await cl.Message(f"❌ Error processing image: {str(e)}").send()

def check_empty_spaces(shelf_img):
    try:
        shelf_hsv = cv2.cvtColor(shelf_img, cv2.COLOR_BGR2HSV)
        report_lines = []
        total_spots = len(planogram_data)
        empty_spots = 0

        debug_lines = []  # For debugging output

        for item in planogram_data:
            x, y, w, h = item['x'], item['y'], item['width'], item['height']
            h_img, w_img = shelf_hsv.shape[:2]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            if w <= 0 or h <= 0:
                logger.warning(f"Invalid dimensions for {item['name']}: x={x}, y={y}, w={w}, h={h}")
                continue

            shelf_crop = shelf_hsv[y:y+h, x:x+w]
            avg_brightness = np.mean(shelf_crop[:, :, 2])
            avg_saturation = np.mean(shelf_crop[:, :, 1])

            debug_line = f"{item['name']} (SKU {item['sku']}): Brightness={avg_brightness:.1f}, Saturation={avg_saturation:.1f}"

            # New rule: low brightness and moderate/low saturation
            if avg_brightness < 70 and avg_saturation < 160:
                empty_spots += 1
                report_lines.append(f"❌ {item['name']} (SKU {item['sku']}) is missing!")
                debug_line += " <-- Detected as empty"
            debug_lines.append(debug_line)

        for line in debug_lines:
            logger.info(line)

        if not report_lines:
            return "✅ All spots look filled! Nice work!"
        else:
            summary = f"\n\n📊 Summary: {empty_spots} out of {total_spots} spots are empty ({empty_spots/total_spots*100:.1f}%)"
            return "\n".join(report_lines) + summary

    except Exception as e:
        logger.error(f"Error in check_empty_spaces: {str(e)}")
        raise