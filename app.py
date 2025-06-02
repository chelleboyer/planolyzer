import chainlit as cl
import cv2
import numpy as np
import json
import os
import logging
from pathlib import Path
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import hf_hub_download

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use relative paths
BASE_DIR = Path(__file__).parent
PLANOGRAM_JSON = BASE_DIR / 'data' / 'product_positions_adjusted_v10.json'
PLANOGRAM_IMAGE = BASE_DIR / 'data' / 'shelf_overlay_adjusted_v10.jpg'
REFERENCE_DIR = BASE_DIR / 'data' / 'planogram001'

# Initialize CLIP model and processor with a smaller model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch16"  # Smaller model variant
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

# Dictionary to store reference images
reference_images = {}

def compress_image(image, max_size=(800, 800)):
    """Compress image while maintaining aspect ratio."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Calculate new dimensions
    ratio = min(max_size[0]/image.size[0], max_size[1]/image.size[1])
    new_size = tuple(int(dim * ratio) for dim in image.size)
    
    # Resize image
    return image.resize(new_size, Image.Resampling.LANCZOS)

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

# Load and compress planogram image
try:
    planogram_image = cv2.imread(str(PLANOGRAM_IMAGE))
    validate_image(planogram_image, "planogram image")
    planogram_image = compress_image(planogram_image)
    logger.info(f"Successfully loaded and compressed planogram image from {PLANOGRAM_IMAGE}")
except Exception as e:
    logger.error(f"Error loading planogram image: {str(e)}")
    raise

def load_reference_images():
    """Load all reference product images."""
    try:
        for img_path in REFERENCE_DIR.glob('*.png'):
            if img_path.name == 'empty-space.png' or img_path.name == 'planogram.png':
                continue
            img = Image.open(img_path)
            reference_images[img_path.stem] = compress_image(img)
            logger.info(f"Loaded reference image: {img_path.stem}")
        logger.info(f"Loaded {len(reference_images)} reference images")
        logger.info(f"Available reference images: {list(reference_images.keys())}")
    except Exception as e:
        logger.error(f"Error loading reference images: {str(e)}")
        raise

def get_expected_image_name(item_name):
    """Convert item name to expected image filename."""
    # Remove numbers and spaces, convert to uppercase
    base_name = ''.join(c for c in item_name.split()[0] if not c.isdigit()).upper()
    logger.info(f"Converting item name '{item_name}' to expected image name '{base_name}'")
    return base_name

def compare_with_reference(crop_img, expected_name):
    """Compare a crop with its expected reference image using CLIP."""
    try:
        # Convert OpenCV image to PIL Image and compress
        if isinstance(crop_img, np.ndarray):
            crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        crop_img = compress_image(crop_img)
        
        # Get expected reference image
        expected_img = reference_images.get(expected_name)
        if expected_img is None:
            logger.warning(f"No reference image found for {expected_name}")
            return None
        
        logger.info(f"Comparing with reference image: {expected_name}")
        
        # Process images with CLIP
        inputs = processor(
            images=[expected_img, crop_img],
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Get image features
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
        
        # Calculate similarity
        similarity = torch.nn.functional.cosine_similarity(
            image_features[0].unsqueeze(0),
            image_features[1].unsqueeze(0)
        ).item()
        
        logger.info(f"Similarity score for {expected_name}: {similarity:.3f}")
        return similarity
    except Exception as e:
        logger.error(f"Error in reference comparison: {str(e)}")
        return None

def load_reference_image():
    """Load and preprocess the reference planogram image."""
    try:
        ref_img = Image.open(REFERENCE_DIR / 'planogram.png')
        if ref_img is None:
            raise FileNotFoundError(f"Reference image {REFERENCE_DIR / 'planogram.png'} not found")
        return compress_image(ref_img)
    except Exception as e:
        print(f"Error loading reference image: {e}")
        return None

def compare_images(ref_img, uploaded_img):
    """Compare the uploaded image with the reference image using CLIP."""
    try:
        # Convert OpenCV image to PIL Image and compress
        if isinstance(uploaded_img, np.ndarray):
            uploaded_img = Image.fromarray(cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2RGB))
        uploaded_img = compress_image(uploaded_img)
        
        # Process images with CLIP
        inputs = processor(
            images=[ref_img, uploaded_img],
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Get image features
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
        
        # Calculate similarity
        similarity = torch.nn.functional.cosine_similarity(
            image_features[0].unsqueeze(0),
            image_features[1].unsqueeze(0)
        ).item()
        
        # Calculate difference percentage (inverse of similarity)
        diff_percentage = (1 - similarity) * 100
        
        return {
            'similarity_score': similarity,
            'difference_percentage': diff_percentage,
            'is_similar': similarity > 0.85
        }
    except Exception as e:
        logger.error(f"Error in CLIP comparison: {str(e)}")
        raise

@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    # Load reference images
    load_reference_images()
    
    ref_img = load_reference_image()
    if ref_img is None:
        await cl.Message(
            content="Error: Reference planogram image not found. Please ensure planogram.png exists in the project directory."
        ).send()
        return
    
    # Create a welcome message with instructions
    welcome_msg = """
# Welcome to Planolyzer! 🛍️

## Quick Start:
1. Download the test image below
2. Upload it back to see how the system works
3. Try creating your own test image by:
   - Adding empty spaces to the reference planogram
   - Moving items to wrong positions
"""
    
    # Send welcome message
    await cl.Message(content=welcome_msg).send()
    
    # Send reference planogram
    await cl.Message(
        content="## Reference Planogram:",
        elements=[
            cl.Image(
                name="planogram",
                path=str(REFERENCE_DIR / 'planogram.png'),
                display="inline",
                size="medium"
            ),
            cl.File(
                name="planogram.png",
                path=str(REFERENCE_DIR / 'planogram.png'),
                display="inline"
            ),
            cl.Image(
                name="empty_space",
                path=str(REFERENCE_DIR / 'empty-space.png'),
                display="inline",
                size="small"
            ),
            cl.File(
                name="empty-space.png",
                path=str(REFERENCE_DIR / 'empty-space.png'),
                display="inline"
            )
        ]
    ).send()
    
    # Send test image
    await cl.Message(
        content="## Test Image:",
        elements=[
            cl.Image(
                name="test_shelf",
                path=str(BASE_DIR / 'data' / 'test_shelf_image_cig_003.png'),
                display="inline",
                size="medium"
            ),
            cl.File(
                name="test_shelf_image_cig_003.png",
                path=str(BASE_DIR / 'data' / 'test_shelf_image_cig_003.png'),
                display="inline"
            )
        ]
    ).send()
    
    # Send additional instructions
    await cl.Message(
        content="Try downloading and uploading the test image first to see how the system works!"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and image uploads."""
    # If there are no elements and no message content, do nothing
    if not message.elements and not message.content:
        return

    # If there are elements, process them regardless of message content
    if message.elements:
        # Get the uploaded image
        uploaded_image = message.elements[0]
        if not uploaded_image.mime.startswith('image/'):
            await cl.Message(
                content="Please upload a valid image file."
            ).send()
            return

        try:
            # Show 'just a moment' message
            await cl.Message(content="⏳ Just a moment while I analyze your image...").send()
            # Show spinner/working message
            await cl.Message(content="🔄 Analyzing image, please wait...").send()
            # Convert uploaded image to OpenCV format
            img_path = uploaded_image.path
            uploaded_img = cv2.imread(img_path)
            if uploaded_img is None:
                await cl.Message(content="Error: Could not read the uploaded image. Please try again with a different image.").send()
                return

            # Load reference image
            ref_img = load_reference_image()
            if ref_img is None:
                await cl.Message(content="Error: Reference planogram image not found.").send()
                return

            # Compare images using CLIP
            comparison_result = compare_images(ref_img, uploaded_img)

            # Prepare response
            if comparison_result['is_similar']:
                # Image is accepted, proceed with empty space analysis
                await cl.Message(content=f"✅ Image accepted! Similarity score: {comparison_result['similarity_score']:.2f}\n\nAnalyzing empty spaces...").send()
                # Perform empty space analysis
                analysis_result = check_empty_spaces(uploaded_img)
                await cl.Message(content=analysis_result).send()
            else:
                # Image is rejected
                response = f"❌ No go! Image rejected.\n"
                response += f"Similarity score: {comparison_result['similarity_score']:.2f}\n"
                response += f"Difference percentage: {comparison_result['difference_percentage']:.2f}%\n\n"
                response += "Please upload a different image that better matches the reference planogram."
                await cl.Message(content=response).send()

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            await cl.Message(
                content=f"Error processing image: {str(e)}"
            ).send()
    else:
        # Handle text-only messages
        await cl.Message(
            content="👋 Hi! I'm here to help you analyze planogram images. Please upload an image to get started. You can use the test image provided above to try out the system!"
        ).send()

def identify_wrong_item(crop_img):
    best_score = -1
    best_name = None
    for ref_name, ref_img in reference_images.items():
        score = compare_with_reference(crop_img, ref_name)
        if score is not None and score > best_score:
            best_score = score
            best_name = ref_name
    return best_name, best_score

def identify_by_text_prompt(crop_img, product_names):
    # Convert crop to PIL and compress
    if isinstance(crop_img, np.ndarray):
        crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    crop_img = compress_image(crop_img)
    # Encode image
    image_inputs = processor(images=crop_img, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
    # Encode text prompts
    prompts = [f"{name} cigarette pack" for name in product_names]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    # Compute similarities
    similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()
    best_idx = similarities.argmax()
    return product_names[best_idx], similarities[best_idx]

def check_empty_spaces(shelf_img):
    try:
        shelf_hsv = cv2.cvtColor(shelf_img, cv2.COLOR_BGR2HSV)
        report_lines = []
        total_spots = len(planogram_data)
        empty_spots = 0
        wrong_items = 0

        debug_lines = []  # For debugging output
        # Prepare product names for text prompt matching
        product_names = [Path(item['image_file']).stem.replace('-', ' ').replace('_', ' ') for item in planogram_data]

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

            # Check if spot is empty
            if avg_brightness < 70 and avg_saturation < 160:
                empty_spots += 1
                report_lines.append(f"❌ {item['name']} (SKU {item['sku']}) is missing!")
                debug_line += " <-- Detected as empty"
            else:
                # If not empty, check if it's the correct item
                image_file_stem = Path(item['image_file']).stem
                similarity = compare_with_reference(shelf_img[y:y+h, x:x+w], image_file_stem)
                
                if similarity is not None:
                    logger.info(f"Item: {item['name']}, Expected: {image_file_stem}, Similarity: {similarity:.3f}")
                    if similarity < 0.80:
                        wrong_items += 1
                        # Identify the most likely wrong item (image-to-image)
                        best_name, best_score = identify_wrong_item(shelf_img[y:y+h, x:x+w])
                        # Identify by text prompt
                        text_best_name, text_best_score = identify_by_text_prompt(shelf_img[y:y+h, x:x+w], product_names)
                        report_lines.append(
                            f"⚠️ {item['name']} (SKU {item['sku']}) appears to be in the wrong spot! (Image similarity: {similarity:.2f})\n"
                            f"   Most likely (image): {best_name} (Similarity: {best_score:.2f})\n"
                            f"   Most likely (text): {text_best_name} (Similarity: {text_best_score:.2f})"
                        )
                        debug_line += f" <-- Wrong item detected (Image: {best_name}, {best_score:.2f}; Text: {text_best_name}, {text_best_score:.2f})"
            
            debug_lines.append(debug_line)

        for line in debug_lines:
            logger.info(line)

        if not report_lines:
            return "✅ All spots look correct! Nice work!"
        else:
            summary = f"\n\n📊 Summary:\n"
            if empty_spots > 0:
                summary += f"- {empty_spots} out of {total_spots} spots are empty ({empty_spots/total_spots*100:.1f}%)\n"
            if wrong_items > 0:
                summary += f"- {wrong_items} out of {total_spots} spots have wrong items ({wrong_items/total_spots*100:.1f}%)"
            return "\n".join(report_lines) + summary
    except Exception as e:
        logger.error(f"Error in check_empty_spaces: {str(e)}")
        raise