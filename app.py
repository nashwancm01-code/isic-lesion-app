import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndi
from skimage import measure
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, binary_closing, remove_small_objects
from skimage.measure import regionprops
from skimage.metrics import mean_squared_error

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="ISIC Skin Lesion Analysis", layout="wide")
st.title("🔬 Analisis & Segmentasi Citra Lesi Kulit (ISIC)")
st.markdown("Aplikasi pemrosesan citra medis untuk mengekstraksi fitur dari lesi kulit.")

# ==========================================
# FUNGSI-FUNGSI BANTUAN
# ==========================================
def mean_filter_manual(image):
    tinggi, lebar = image.shape
    output = np.zeros((tinggi, lebar), dtype=np.uint8)
    for i in range(1, tinggi - 1):
        for j in range(1, lebar - 1):
            jumlah = 0
            for y in range(-1, 2):
                for x in range(-1, 2):
                    jumlah += int(image[i + y, j + x])
            output[i, j] = jumlah // 9
    return output

def gaussian_filter_manual(image):
    tinggi, lebar = image.shape
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float) / 16
    output = np.zeros((tinggi, lebar), dtype=np.uint8)
    for i in range(1, tinggi - 1):
        for j in range(1, lebar - 1):
            jumlah = 0.0
            for y in range(-1, 2):
                for x in range(-1, 2):
                    jumlah += (image[i + y, j + x] * kernel[y + 1, x + 1])
            output[i, j] = np.clip(round(jumlah), 0, 255)
    return output

def hitung_psnr_manual(mse, max_pixel=255):
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_pixel ** 2) / mse)

# ==========================================
# SIDEBAR: UPLOAD GAMBAR
# ==========================================
st.sidebar.header("📥 Input Data")
uploaded_files = st.sidebar.file_uploader("Upload Citra Lesi (Maks. 3, JPG/PNG)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    # Membatasi maksimal 3 gambar
    if len(uploaded_files) > 3:
        st.sidebar.warning("Anda mengunggah lebih dari 3 gambar. Hanya 3 gambar pertama yang akan diproses.")
        uploaded_files = uploaded_files[:3]

    st.sidebar.success(f"{len(uploaded_files)} Gambar berhasil dimuat!")

    # Looping untuk setiap gambar yang diupload
    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"---")
        st.markdown(f"## 🖼️ HASIL ANALISIS GAMBAR {idx + 1}: `{uploaded_file.name}`")
        
        # Membaca gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        st.sidebar.image(image_rgb, caption=f"Citra {idx+1}: {uploaded_file.name}", use_column_width=True)

        # ==========================================
        # 1. RGB CHANNEL & HISTOGRAM
        # ==========================================
        st.subheader("1. Ekstraksi Channel Warna & Histogram")
        
        r, g, b = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]
        
        fig1, axes1 = plt.subplots(1, 4, figsize=(16, 4))
        channels = [(image_rgb, None, None, 'Image Asli (RGB)'), 
                    (r, '#e74c3c', 'Red', 'Red Channel & Histogram'), 
                    (g, '#2ecc71', 'Green', 'Green Channel & Histogram'), 
                    (b, '#3498db', 'Blue', 'Blue Channel & Histogram')]

        for col_idx, (data, c_color, c_name, title) in enumerate(channels):
            ax = axes1[col_idx]
            ax.axis('off')
            
            if col_idx == 0:
                ax.imshow(data)
                ax.set_title(title, fontsize=10, fontweight='bold', pad=12)
            else:
                sub_img_ax = ax.inset_axes([0.0, 0.35, 1.0, 0.65])
                sub_img_ax.imshow(data, cmap='gray')
                sub_img_ax.axis('off')
                
                sub_hist_ax = ax.inset_axes([0.0, 0.0, 1.0, 0.30])
                hist_vals, bins = np.histogram(data.ravel(), bins=256, range=[0, 256])
                sub_hist_ax.bar(range(256), hist_vals, width=1.0, color=c_color, alpha=0.8)
                sub_hist_ax.set_xlim([0, 256])
                sub_hist_ax.tick_params(axis='both', labelsize=6)
                
                ax.set_title(title, fontsize=10, fontweight='bold', pad=12)

        st.pyplot(fig1)

        # ==========================================
        # 2. ENHANCEMENT (CONTRAST STRETCHING)
        # ==========================================
        st.subheader("2. Enhancement (Contrast Stretching)")
        
        g_original = g.astype(np.float32)
        c_min, d_max = np.min(g_original), np.max(g_original)
        
        if d_max != c_min:
            g_enhanced = ((g_original - c_min) * (255 - 0) / (d_max - c_min) + 0)
        else:
            g_enhanced = g_original.copy()
            
        g_enhanced = np.clip(g_enhanced, 0, 255).astype(np.uint8)

        fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
        axes2[0].imshow(image_rgb); axes2[0].set_title("Image Asli (RGB)")
        axes2[1].imshow(g_original.astype(np.uint8), cmap='gray'); axes2[1].set_title("Green Channel (Asli)")
        axes2[2].imshow(g_enhanced, cmap='gray'); axes2[2].set_title("Green Channel (Enhanced)")
        for ax in axes2: ax.axis('off')
        st.pyplot(fig2)

        # ==========================================
        # 3. FILTERING & PERHITUNGAN METRIK
        # ==========================================
        st.subheader("3. Evaluasi Filtering & Metrik (PSNR & MSE)")
        
        with st.spinner(f"Memproses Filter untuk Gambar {idx+1}..."):
            img_mean = mean_filter_manual(g_enhanced)
            img_gauss = gaussian_filter_manual(g_enhanced)
            img_median = ndi.median_filter(g_enhanced, size=3)
            img_min = ndi.minimum_filter(g_enhanced, size=3)
            img_max = ndi.maximum_filter(g_enhanced, size=3)

        filtered_dict = {
            "Mean": img_mean, "Gaussian": img_gauss, 
            "Median": img_median, "Min": img_min, "Max": img_max
        }

        fig3, axes3 = plt.subplots(1, 6, figsize=(18, 4))
        plot_filters = [("Green Enhanced", g_enhanced)] + list(filtered_dict.items())
        
        for p_idx, (name, img_f) in enumerate(plot_filters):
            axes3[p_idx].imshow(img_f, cmap='gray')
            axes3[p_idx].set_title(name, fontsize=10, fontweight='bold')
            axes3[p_idx].axis('off')
        st.pyplot(fig3)

        # Tabel Evaluasi
        eval_results = []
        for name, img_f in filtered_dict.items():
            mse_val = mean_squared_error(g_enhanced, img_f)
            psnr_val = hitung_psnr_manual(mse_val)
            eval_results.append({"Filter": name, "MSE": mse_val, "PSNR (dB)": psnr_val})
        
        st.table(pd.DataFrame(eval_results).set_index("Filter"))

        # ==========================================
        # 4. SEGMENTASI MORFOLOGI
        # ==========================================
        st.subheader("4. Segmentasi & Operasi Morfologi")
        
        nilai_otsu = threshold_otsu(img_median)
        otsu_thresh = img_median > nilai_otsu
        mask_binary = np.logical_not(otsu_thresh)
        mask_awal = mask_binary.astype(np.uint8) * 255

        kernel_morph = np.ones((3, 3), dtype=bool)
        hasil_erosion = ndi.binary_erosion(mask_binary, structure=kernel_morph).astype(np.uint8) * 255
        hasil_dilation = ndi.binary_dilation(mask_binary, structure=kernel_morph).astype(np.uint8) * 255
        hasil_opening = ndi.binary_opening(mask_binary, structure=kernel_morph).astype(np.uint8) * 255
        hasil_closing = ndi.binary_closing(mask_binary, structure=kernel_morph).astype(np.uint8) * 255

        fig4, axes4 = plt.subplots(1, 5, figsize=(16, 4))
        morph_titles = ["Mask Otsu", "Erosion", "Dilation", "Opening", "Closing"]
        morph_imgs = [mask_awal, hasil_erosion, hasil_dilation, hasil_opening, hasil_closing]
        
        for m_idx, (name, img_m) in enumerate(zip(morph_titles, morph_imgs)):
            axes4[m_idx].imshow(img_m, cmap='gray')
            axes4[m_idx].set_title(name, fontweight='bold')
            axes4[m_idx].axis('off')
        st.pyplot(fig4)

        # ==========================================
        # 5. LABELING & EKSTRAKSI FITUR
        # ==========================================
        st.subheader("5. Ekstraksi Fitur Geometri (Labeling)")
        
        opened_mask = binary_opening(mask_binary, footprint=np.array([[False, True, False], [True, True, True], [False, True, False]], dtype=bool))
        cleaned_binary = remove_small_objects(opened_mask, min_size=30)
        labeled = measure.label(cleaned_binary, connectivity=2)
        
        if labeled.max() > 0:
            regions = regionprops(labeled)
            region_terbesar = max(regions, key=lambda region: region.area)
            cleaned_binary = (labeled == region_terbesar.label)
            
        final_binary = ndi.binary_fill_holes(cleaned_binary)
        final_binary = binary_closing(final_binary, footprint=np.array([[False, True, False], [True, True, True], [False, True, False]], dtype=bool))
        final_binary = ndi.binary_fill_holes(final_binary)
        
        labeled_lesions = measure.label(final_binary > 0, connectivity=2)
        nlabels = labeled_lesions.max()

        col1, col2 = st.columns([1, 2])
        with col1:
            fig5, ax5 = plt.subplots(figsize=(4, 4))
            ax5.imshow(labeled_lesions, cmap="nipy_spectral")
            ax5.set_title(f"Terdeteksi {nlabels} Region Lesi")
            ax5.axis("off")
            st.pyplot(fig5)
            
        with col2:
            properties = regionprops(labeled_lesions)
            fitur_list = []
            for prop in properties:
                fitur_list.append({
                    "label_id": prop.label,
                    "area": prop.area,
                    "centroid_row": prop.centroid[0],
                    "centroid_col": prop.centroid[1],
                    "major_axis_length": prop.major_axis_length,
                    "minor_axis_length": prop.minor_axis_length,
                    "eccentricity": prop.eccentricity,
                    "orientation": prop.orientation
                })
                
            df_fitur = pd.DataFrame(fitur_list)
            st.write(f"Tabel Ekstraksi Fitur - Gambar {idx+1}:")
            st.dataframe(df_fitur, use_container_width=True)
            
            # Tombol Download CSV (menggunakan key unik per gambar)
            csv = df_fitur.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"⬇️ Download Data Fitur Gambar {idx+1} (CSV)",
                data=csv,
                file_name=f"fitur_lesi_{uploaded_file.name}.csv",
                mime="text/csv",
                key=f"download_csv_{idx}"
            )

else:
    st.info("Silakan upload hingga 3 gambar citra medis kulit dari panel di sebelah kiri untuk memulai analisis.")
