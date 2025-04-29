# app.py

import streamlit as st
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_ubyte, exposure
from skimage.filters import sobel, laplace, unsharp_mask
from skimage.feature import canny
import os

# Funzione salvataggio immagini
def salva_fig(path_output, fig=None):
    os.makedirs(os.path.dirname(path_output), exist_ok=True)
    if fig is None:
        plt.savefig(path_output, bbox_inches='tight', dpi=300)
    else:
        fig.savefig(path_output, bbox_inches='tight', dpi=300)

# Titolo interfaccia
st.set_page_config(layout="wide")
st.title("🧠 Analisi Batch di Immagini DICOM per Supporto Clinico")

# Upload multiplo file DICOM
uploaded_files = st.file_uploader("📤 Carica uno o più file DICOM", type=["dcm"], accept_multiple_files=True)

if uploaded_files:
    images = []
    filenames = []
    for uploaded_file in uploaded_files:
        dcm = pydicom.dcmread(uploaded_file)
        img = dcm.pixel_array
        images.append(img)
        filenames.append(os.path.splitext(uploaded_file.name)[0])

    # Checkbox per selezionare trasformazioni
    st.sidebar.header("🔧 Seleziona Trasformazioni da Applicare")
    do_entropy = st.sidebar.checkbox("Mappa di Entropia Locale")
    do_edges = st.sidebar.checkbox("Edge Detection: Sobel, Laplaciano, Canny")
    do_eq = st.sidebar.checkbox("Equalizzazione Istogramma")
    do_sharp = st.sidebar.checkbox("Filtro Unsharp Mask (Velvet-like)")
    do_overlay = st.sidebar.checkbox("Overlay dei bordi Canny")

    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    # Pulsanti di navigazione
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("⬅️ Previous"):
            if st.session_state.current_index > 0:
                st.session_state.current_index -= 1
    with col3:
        if st.button("Next ➡️"):
            if st.session_state.current_index < len(images) - 1:
                st.session_state.current_index += 1

    img = images[st.session_state.current_index]
    base_filename = filenames[st.session_state.current_index]
    img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
    img_uint8 = img_as_ubyte(img_norm)

    st.subheader(f"🖼️ Immagine {st.session_state.current_index + 1} di {len(images)} - {base_filename}")
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img, cmap='gray')
    ax.set_title("Immagine DICOM Originale")
    ax.axis('off')
    st.pyplot(fig)

    output_dir = "transformations"
    os.makedirs(output_dir, exist_ok=True)

    # Applica trasformazioni selezionate
    if do_entropy:
        entropia = entropy(img_uint8, disk(5))
        plt.figure(figsize=(6,6))
        plt.imshow(entropia, cmap='inferno')
        plt.title('Mappa di Entropia Locale')
        plt.axis('off')
        plt.colorbar()
        salva_fig(f'{output_dir}/{base_filename}_entropy_map.png')
        st.pyplot(plt)
        plt.close()

    if do_edges:
        edges_sobel = sobel(img)
        edges_laplace = laplace(img)
        edges_canny = canny(img_norm, sigma=1)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(edges_sobel, cmap='gray')
        axs[0].set_title('Bordi - Sobel')
        axs[0].axis('off')
        axs[1].imshow(edges_laplace, cmap='gray')
        axs[1].set_title('Bordi - Laplaciano')
        axs[1].axis('off')
        axs[2].imshow(edges_canny, cmap='gray')
        axs[2].set_title('Bordi - Canny')
        axs[2].axis('off')
        plt.tight_layout()
        salva_fig(f'{output_dir}/{base_filename}_edges_all.png', fig)
        st.pyplot(fig)
        plt.close()

    if do_eq:
        img_eq = exposure.equalize_hist(img_norm)
        plt.figure(figsize=(6,6))
        plt.imshow(img_eq, cmap='gray')
        plt.title('Equalizzazione Istogramma')
        plt.axis('off')
        salva_fig(f'{output_dir}/{base_filename}_equalized_histogram.png')
        st.pyplot(plt)
        plt.close()

    if do_sharp:
        img_sharp = unsharp_mask(img_norm, radius=1.0, amount=1.5)
        plt.figure(figsize=(6,6))
        plt.imshow(img_sharp, cmap='gray')
        plt.title('Filtro Unsharp Mask (Velvet-like)')
        plt.axis('off')
        salva_fig(f'{output_dir}/{base_filename}_unsharp_mask.png')
        st.pyplot(plt)
        plt.close()

    if do_overlay:
        if 'edges_canny' not in locals():
            edges_canny = canny(img_norm, sigma=1)
        overlay = np.stack([img_norm]*3, axis=-1)
        overlay[edges_canny, 0] = 1
        overlay[edges_canny, 1] = 0
        overlay[edges_canny, 2] = 0
        plt.figure(figsize=(6,6))
        plt.imshow(overlay)
        plt.title('Overlay Bordi (Canny) su Immagine Originale')
        plt.axis('off')
        salva_fig(f'{output_dir}/{base_filename}_overlay_edges_on_original.png')
        st.pyplot(plt)
        plt.close()

else:
    st.info("📤 Carica uno o più file DICOM per iniziare.")
