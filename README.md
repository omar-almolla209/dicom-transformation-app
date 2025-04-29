
# 🧠 DICOM Image Enhancement App for Medical Interpretation

This application provides a user-friendly, browser-based interface for uploading and transforming DICOM images. It is designed to assist physicians in the visual interpretation of subtle or low-contrast structures using a set of medical image processing techniques.

The app is built using **Streamlit** and works entirely in the browser. Users can upload `.DCM` files and interactively apply different transformations to extract features such as edges, texture, and contrast that might be missed in the raw scan.

---

## 🔧 How It Works

1. Upload a `.DCM` image (DICOM format).
2. Select one or more transformations from the sidebar.
3. View the processed images in real-time.
4. Each output is also saved to the `transformations/` directory for later reference.

---

## 🔍 Transformations Explained

### 1. Local Entropy Map
- **Purpose**: Highlights regions with high local variation (e.g. lesions, heterogeneous tissue).
- **Theory**: Shannon entropy is computed over a sliding window. High entropy = complex local pixel structure.

### 2. Edge Detection (Sobel, Laplacian, Canny)
- **Purpose**: Detects sharp changes in intensity, often corresponding to anatomical borders.
- **Details**:
  - **Sobel**: Computes intensity gradients (horizontal/vertical).
  - **Laplacian**: Second derivative filter for high-frequency detail.
  - **Canny**: Advanced edge detection combining noise reduction and gradient thresholds.

### 3. Histogram Equalization
- **Purpose**: Enhances visibility in low-contrast images (common in soft tissue scans).
- **Theory**: Redistributes pixel intensities to span the full dynamic range uniformly.

### 4. Unsharp Mask (Velvet-like Enhancement)
- **Purpose**: Gently sharpens important details while preserving soft gradients.
- **Theory**: Subtracts a blurred version from the original image, boosting local contrast.

### 5. Edge Overlay (Canny + Original)
- **Purpose**: Superimposes detected edges onto the original grayscale image to contextualize the highlighted structures.
- **Theory**: Canny edges are colored (e.g., red) and combined with the normalized grayscale image in RGB.

---

## 👩‍⚕️ Why This Is Useful for Clinicians

- Reveals **subtle visual features** and **low-contrast areas** often missed on standard viewing tools.
- Enhances **structural recognition** without altering the raw DICOM data.
- Useful for:
  - Diagnostic assistance
  - Pre-surgical planning
  - Teaching and radiology education
  - Image comparison between modalities or patients

---

## 🚀 Deployment

You can deploy this app for free using **Streamlit Community Cloud**:

1. Push this repository to GitHub.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud).
3. Sign in with GitHub and click **"New app"**.
4. Select your repository and choose `app.py` as the entry point.
5. Done! You’ll get a public link like:

