## 🛍️ **Planolyzer Application Overview**

### **Purpose**
Planolyzer is an AI-powered tool designed to analyze retail shelf images and automatically check planogram compliance. It detects:
- **Empty spaces** (missing products)
- **Wrong items** (misplaced products)
- **Brand mismatches** (using OCR and AI vision)

---

### **How It Works**

#### 1. **Startup & Reference Data**
- On launch, Planolyzer loads:
  - A **planogram JSON** file describing the expected products, their positions, and the reference image filenames.
  - **Reference product images** for each SKU from a directory.
  - The **reference planogram image** (the “ideal” shelf).

#### 2. **User Interaction**
- The user is greeted with instructions and sample images.
- The user uploads a photo of a retail shelf.

#### 3. **Image Validation**
- The app checks that the uploaded image is valid and readable.

#### 4. **Planogram Comparison**
- The uploaded image is compared to the reference planogram image using the CLIP model to ensure it’s a valid shelf photo.
- If the similarity is too low, the user is asked to upload a better image.

#### 5. **Spot-by-Spot Analysis**
For each product spot defined in the planogram:
- The app crops the corresponding region from the uploaded image.
- It checks if the spot is **empty** (using brightness and saturation thresholds).
- If not empty, it:
  - **Compares the crop to the expected reference image** using CLIP (image-to-image similarity).
  - **Compares the crop to all reference images** to identify the most likely product if the item is wrong.
  - **Uses CLIP text prompt matching** (image-to-text) to further identify the product.
  - **Runs OCR** (EasyOCR) on the crop to extract any visible brand name or text.
  - **Fuzzy matches** the OCR result to known brands for robust detection.

#### 6. **Reporting**
- The app generates a detailed report for the user, including:
  - Which spots are empty (❌)
  - Which spots have the wrong item (⚠️), what the item most likely is, and what text was detected
  - A summary of empty and wrong spots as counts and percentages

#### 7. **User Feedback**
- The app provides real-time feedback with “Just a moment…” and “Analyzing…” messages while processing.
- Results are displayed as soon as analysis is complete.

---

### **Key Technologies Used**
- **CLIP (Contrastive Language-Image Pretraining):** For image and text-based product recognition.
- **EasyOCR:** For reading brand names and text from product images.
- **Chainlit:** For the interactive chat/web interface.
- **OpenCV & PIL:** For image processing and cropping.
