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

st.title("🧠 Analisi Immagini DICOM – Supporto Visivo per Medici")



# Upload file DICOM

uploaded_files = st.file_uploader("📤 Carica uno o più file DICOM", type=None, accept_multiple_files=True)



if uploaded_file:

    dcm = pydicom.dcmread(uploaded_file)

    img = dcm.pixel_array

    img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))

    img_uint8 = img_as_ubyte(img_norm)



    st.subheader("📸 Immagine Originale")

    fig, ax = plt.subplots(figsize=(6,6))

    ax.imshow(img, cmap='gray')

    ax.set_title("Immagine DICOM Originale")

    ax.axis('off')

    st.pyplot(fig)





    # Checkbox per ogni trasformazione

    col1, col2 = st.columns(2)

    with col1:

        do_entropy = st.checkbox("Mappa di Entropia Locale")

        do_edges = st.checkbox("Edge Detection: Sobel, Laplaciano, Canny")

        do_overlay = st.checkbox("Overlay dei bordi Canny")



    with col2:

        do_eq = st.checkbox("Equalizzazione Istogramma")

        do_sharp = st.checkbox("Filtro Unsharp Mask (Velvet-like)")



    base_filename = os.path.splitext(uploaded_file.name)[0]

    output_dir = "transformations"

    os.makedirs(output_dir, exist_ok=True)



    # Esecuzione trasformazioni selezionate

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
