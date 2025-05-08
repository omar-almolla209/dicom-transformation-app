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

# Upload file DICOM - MODIFICATO QUI per includere 'dmc'
uploaded_files = st.file_uploader("📤 Carica uno o più file DICOM", type=None, accept_multiple_files=True)

if uploaded_file:
    try:
        dcm = pydicom.dcmread(uploaded_file)
        img = dcm.pixel_array

        # Controllo per immagini multi-frame (es. alcune sequenze cine)
        # Questo è un esempio base, potresti aver bisogno di logica più sofisticata
        # per selezionare un frame specifico o processarli tutti.
        if img.ndim > 2:
            st.warning(f"L'immagine DICOM ha {img.ndim} dimensioni (probabilmente multi-frame). Verrà visualizzato il primo frame.")
            # Se è multi-frame, prendi il primo frame per le elaborazioni successive
            if img.ndim == 3: # (frames, rows, cols)
                img = img[0, :, :]
            elif img.ndim == 4: # (frames, rows, cols, channels)
                 img = img[0, :, :, 0] # Prendi il primo frame, primo canale (se scala di grigi)
            else:
                st.error("Formato immagine multi-frame non supportato per l'elaborazione automatica. Controlla i dati.")
                st.stop()


        # Normalizzazione e conversione
        # Aggiunto un controllo per evitare divisione per zero se l'immagine è costante
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val == min_val:
            img_norm = np.zeros_like(img, dtype=np.float32) # o np.ones_like se preferisci
        else:
            img_norm = (img - min_val) / (max_val - min_val)
        
        img_uint8 = img_as_ubyte(img_norm)

        st.subheader("📸 Immagine Originale")
        fig_orig, ax_orig = plt.subplots(figsize=(6,6))
        ax_orig.imshow(img, cmap='gray')
        ax_orig.set_title("Immagine DICOM Originale")
        ax_orig.axis('off')
        st.pyplot(fig_orig)
        plt.close(fig_orig) # Chiudi la figura originale dopo averla mostrata


        # Checkbox per ogni trasformazione
        st.subheader("🛠️ Seleziona le Trasformazioni da Applicare")
        col1, col2 = st.columns(2)
        with col1:
            do_entropy = st.checkbox("Mappa di Entropia Locale")
            do_edges = st.checkbox("Edge Detection: Sobel, Laplaciano, Canny")
            do_overlay = st.checkbox("Overlay dei bordi Canny (su immagine normalizzata)")

        with col2:
            do_eq = st.checkbox("Equalizzazione Istogramma (su immagine normalizzata)")
            do_sharp = st.checkbox("Filtro Unsharp Mask (su immagine normalizzata)")

        base_filename = os.path.splitext(uploaded_file.name)[0]
        output_dir = "transformations_output" # Rinominato per chiarezza, se preferisci
        os.makedirs(output_dir, exist_ok=True)

        # Esecuzione trasformazioni selezionate
        if do_entropy:
            st.subheader("📊 Mappa di Entropia Locale")
            entropia = entropy(img_uint8, disk(5))
            fig_entropy, ax_entropy = plt.subplots(figsize=(6,6))
            im_entropy = ax_entropy.imshow(entropia, cmap='inferno')
            ax_entropy.set_title('Mappa di Entropia Locale')
            ax_entropy.axis('off')
            plt.colorbar(im_entropy, ax=ax_entropy)
            salva_fig(f'{output_dir}/{base_filename}_entropy_map.png', fig_entropy)
            st.pyplot(fig_entropy)
            plt.close(fig_entropy)

        if do_edges:
            st.subheader("📉 Edge Detection")
            # Usa img_norm per Canny come facevi, img originale per Sobel/Laplace
            edges_sobel = sobel(img_norm) # Applicato a img_norm per consistenza con Canny
            edges_laplace = laplace(img_norm) # Applicato a img_norm
            edges_canny = canny(img_norm, sigma=1)
            
            fig_edges, axs_edges = plt.subplots(1, 3, figsize=(18, 6)) # Aumentata leggermente la larghezza
            axs_edges[0].imshow(edges_sobel, cmap='gray')
            axs_edges[0].set_title('Bordi - Sobel')
            axs_edges[0].axis('off')
            axs_edges[1].imshow(edges_laplace, cmap='gray')
            axs_edges[1].set_title('Bordi - Laplaciano')
            axs_edges[1].axis('off')
            axs_edges[2].imshow(edges_canny, cmap='gray')
            axs_edges[2].set_title('Bordi - Canny')
            axs_edges[2].axis('off')
            plt.tight_layout()
            salva_fig(f'{output_dir}/{base_filename}_edges_all.png', fig_edges)
            st.pyplot(fig_edges)
            plt.close(fig_edges)

        if do_eq:
            st.subheader("🎨 Equalizzazione Istogramma")
            img_eq = exposure.equalize_hist(img_norm)
            fig_eq, ax_eq = plt.subplots(figsize=(6,6))
            ax_eq.imshow(img_eq, cmap='gray')
            ax_eq.set_title('Equalizzazione Istogramma')
            ax_eq.axis('off')
            salva_fig(f'{output_dir}/{base_filename}_equalized_histogram.png', fig_eq)
            st.pyplot(fig_eq)
            plt.close(fig_eq)

        if do_sharp:
            st.subheader("✨ Filtro Unsharp Mask (Velvet-like)")
            img_sharp = unsharp_mask(img_norm, radius=1.0, amount=1.5)
            fig_sharp, ax_sharp = plt.subplots(figsize=(6,6))
            ax_sharp.imshow(img_sharp, cmap='gray')
            ax_sharp.set_title('Filtro Unsharp Mask (Velvet-like)')
            ax_sharp.axis('off')
            salva_fig(f'{output_dir}/{base_filename}_unsharp_mask.png', fig_sharp)
            st.pyplot(fig_sharp)
            plt.close(fig_sharp)

        if do_overlay:
            st.subheader("🖼️ Overlay Bordi Canny")
            # Assicurati che edges_canny sia calcolato se non lo è già stato
            # (ad esempio, se l'utente seleziona solo "Overlay" e non "Edge Detection")
            if 'edges_canny' not in locals() or not do_edges: # Aggiunto 'not do_edges' per ricreare se deselezionato
                edges_canny = canny(img_norm, sigma=1)
            
            # Crea un'immagine RGB da img_norm per l'overlay
            overlay_img = np.stack([img_norm]*3, axis=-1)
            
            # Sovrapponi i bordi in rosso
            overlay_img[edges_canny, 0] = 1 # Canale Rosso
            overlay_img[edges_canny, 1] = 0 # Canale Verde
            overlay_img[edges_canny, 2] = 0 # Canale Blu
            
            fig_overlay, ax_overlay = plt.subplots(figsize=(6,6))
            ax_overlay.imshow(overlay_img) # img_norm già normalizzata tra 0 e 1
            ax_overlay.set_title('Overlay Bordi (Canny) su Immagine Normalizzata')
            ax_overlay.axis('off')
            salva_fig(f'{output_dir}/{base_filename}_overlay_edges_on_original.png', fig_overlay)
            st.pyplot(fig_overlay)
            plt.close(fig_overlay)

    except Exception as e:
        st.error(f"Errore durante l'elaborazione del file DICOM: {e}")
        st.error("Assicurati che il file sia un DICOM valido e contenga dati pixel.")

else:
    st.info("Attendo il caricamento di un file DICOM.")
