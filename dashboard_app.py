
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# import plotly.graph_objects as go # Uncomment jika Anda menggunakannya nanti
from sklearn.metrics import confusion_matrix
import pickle
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import os

# --- Konfigurasi Path & Konstanta ---
# GANTI 'ProyekNLPClickbaitFinal' DENGAN NAMA FOLDER UTAMA ANDA DI GOOGLE DRIVE
BASE_PROJECT_PATH_FOR_APP = "/content/drive/MyDrive/ProyekNLPClickbaitFinal" # Sesuaikan jika perlu
BASE_OUTPUT_PATH_FOR_APP = os.path.join(BASE_PROJECT_PATH_FOR_APP, "dashboard_files")

W2V_VECTOR_SIZE_DASH_APP = 100

# --- Cek dan Download NLTK 'punkt' jika belum ada ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    try:
        st.info("Mengunduh resource NLTK 'punkt'...")
        nltk.download('punkt', quiet=True)
        st.success("Resource NLTK 'punkt' berhasil diunduh.")
    except Exception as e_nltk_download_init:
        st.error(f"Gagal mengunduh 'punkt': {e_nltk_download_init}. Harap unduh manual.")
except Exception as e_nltk_find_init:
    st.warning(f"Error saat memeriksa 'punkt': {e_nltk_find_init}")

# --- Fungsi untuk Menyuntikkan CSS Kustom ---
def inject_custom_css_app():
    css = '''
    <style>
        /* --- Global Font & Theme Adjustments --- */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

        html, body, [class*="css"]  { font-family: 'Roboto', sans-serif; }
        .stApp > header { background-color: transparent; }

        h1#dashboard-analisis-klasifikasi-berita-clickbait,
        h1#detail-model-dan-confusion-matrix,
        h1#analisis-kesalahan-prediksi,
        /* h1#analisis-fitur-feature-importance, */ /* Dihapus */
        h1#prediksi-berita-baru,
        h1#analisis-file-csv-unggahan {
            color: #00BFFF;
            text-shadow: 0 0 3px rgba(0, 191, 255, 0.7);
            text-align: center;
            padding-top: 25px;
            padding-bottom: 25px;
            font-weight: 700;
            letter-spacing: 1px;
        }
        h2, h3 {
            color: #C5C6C7; border-bottom: 1px solid #3A3A5E; padding-bottom: 10px; margin-top: 30px;
        }
        section[data-testid="stSidebar"] {
            background-color: #161B22 !important; border-right: 1px solid #30363D !important; padding-top: 1rem !important;
        }
        section[data-testid="stSidebar"] .stRadio > label > div[data-testid="stMarkdownContainer"] > p {
            color: #E0E0E0 !important; font-weight: 500 !important; font-size: 1.05em !important; padding-left: 5px !important;
        }
        section[data-testid="stSidebar"] .stRadio > label {
            padding: 8px 12px !important; border-radius: 6px !important; transition: background-color 0.2s ease-in-out !important;
        }
        section[data-testid="stSidebar"] .stRadio > label:hover { background-color: #21262D !important; }
        /* section[data-testid="stSidebar"] .stImage > img { margin-bottom: 1rem !important; } */ /* Dihapus */
        .stDataFrame, .stTable { border: 1px solid #30363D; border-radius: 6px; }
        .stDataFrame > div > div > table > thead > tr > th { background-color: #161B22; color: #C5C6C7; font-weight: 500; }
        .stMetric {
            background-color: #0D1117; border: 1px solid #30363D; border-radius: 8px; padding: 20px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2); transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }
        .stMetric:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0, 123, 255, 0.25); }
        .stMetric > label { color: #8B949E; font-size: 0.95em; }
        .stMetric > div:nth-child(2) > div { color: #FAFAFA; font-size: 2em; font-weight: 700; }
        .stButton > button {
            border-radius: 6px; border: 1px solid #238636; color: #FAFAFA; background-color: #238636;
            transition: all 0.2s ease-in-out; padding: 8px 18px; font-weight: 500;
        }
        .stButton > button:hover { background-color: #2EA043; border-color: #2EA043; box-shadow: 0 0 8px rgba(46, 160, 67, 0.5); }
        .stFileUploader > div > button {
            border-radius: 6px; border: 1px dashed #30363D; color: #C5C6C7; background-color: #0D1117;
            transition: all 0.2s ease-in-out; padding: 10px 18px; font-weight: 500;
        }
        .stFileUploader > div > button:hover { border-color: #2F81F7; background-color: #161B22; }
        .stSelectbox > div > div[data-baseweb="select"] { background-color: #0D1117; border: 1px solid #30363D; border-radius: 6px; }
        .stSelectbox [data-testid="stMarkdownContainer"] { color: #C5C6C7; }
        .stTextArea textarea {
            background-color: #0D1117; border: 1px solid #30363D; border-radius: 6px; color: #FAFAFA; min-height: 120px; padding: 10px;
        }
        .stTextArea textarea:focus { border-color: #2F81F7; box-shadow: 0 0 0 3px rgba(47, 129, 247, 0.3); }
        .stAlert > div[role="alert"] { border-radius: 6px; border-left-width: 4px; padding: 12px; }
        .stAlert > div[data-baseweb="alert"][class*="Error"] { border-left-color: #F85149; background-color: rgba(248, 81, 73, 0.1); color: #F85149; }
        .stAlert > div[data-baseweb="alert"][class*="Warning"] { border-left-color: #D29922; background-color: rgba(210, 153, 34, 0.1); color: #D29922; }
        .stAlert > div[data-baseweb="alert"][class*="Success"] { border-left-color: #2EA043; background-color: rgba(46, 160, 67, 0.1); color: #2EA043; }
        .stAlert > div[data-baseweb="alert"][class*="Info"] { border-left-color: #2F81F7; background-color: rgba(47, 129, 247, 0.1); color: #2F81F7; }
        ::-webkit-scrollbar { width: 10px; }
        ::-webkit-scrollbar-track { background: #0D1117; }
        ::-webkit-scrollbar-thumb { background: #30363D; border-radius: 5px;}
        ::-webkit-scrollbar-thumb:hover { background: #58A6FF; }
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

# --- Panggil Fungsi CSS dan Konfigurasi Halaman ---
st.set_page_config(
    layout="wide",
    page_title="Dashboard Analytica Clickbait",
    page_icon="🔮"
)
inject_custom_css_app()

# --- Fungsi Ekstraksi Fitur (harus konsisten dengan prepare_dashboard_data.py) ---
def extract_basic_features_live_app(texts_series_input):
    features_list_output = []
    for text_input in texts_series_input:
        text_str_input = str(text_input)
        word_count_output = len(text_str_input.split())
        char_count_output = len(text_str_input)
        features_list_output.append([word_count_output, char_count_output])
    return np.array(features_list_output)

def get_w2v_embedding_live_app(token_list_input, w2v_model_gensim, vector_size_input):
    if not hasattr(w2v_model_gensim, 'wv'):
        return np.zeros(vector_size_input)
    embeddings_list = [w2v_model_gensim.wv[word_token] for word_token in token_list_input if word_token in w2v_model_gensim.wv]
    if not embeddings_list:
        return np.zeros(vector_size_input)
    return np.mean(embeddings_list, axis=0)

# --- Fungsi Pemuatan Data ---
@st.cache_data
def load_csv_data_streamlit(filename_csv):
    path_csv = os.path.join(BASE_OUTPUT_PATH_FOR_APP, filename_csv)
    if not os.path.exists(path_csv):
        st.error(f"File CSV tidak ditemukan: {path_csv}.")
        return pd.DataFrame()
    return pd.read_csv(path_csv)

@st.cache_data
def load_excel_data_streamlit(filename_excel):
    path_excel = os.path.join(BASE_OUTPUT_PATH_FOR_APP, filename_excel)
    if not os.path.exists(path_excel):
        st.error(f"File Excel tidak ditemukan: {path_excel}.")
        return {}
    try:
        xls_file = pd.ExcelFile(path_excel)
        return {sheet_name_xls: xls_file.parse(sheet_name_xls) for sheet_name_xls in xls_file.sheet_names}
    except Exception as e_excel:
        st.error(f"Gagal memuat file Excel '{path_excel}': {e_excel}")
        return {}

@st.cache_resource
def load_pickle_object_streamlit(filename_pickle):
    path_pickle = os.path.join(BASE_OUTPUT_PATH_FOR_APP, filename_pickle)
    if not os.path.exists(path_pickle):
        st.error(f"File Pickle tidak ditemukan: {path_pickle}.")
        return None
    try:
        with open(path_pickle, 'rb') as f_pickle:
            return pickle.load(f_pickle)
    except Exception as e_pickle:
        st.error(f"Gagal memuat file Pickle '{path_pickle}': {e_pickle}")
        return None

@st.cache_resource
def load_gensim_w2v_model_streamlit(filename_gensim="word2vec.model"):
    path_gensim = os.path.join(BASE_OUTPUT_PATH_FOR_APP, filename_gensim)
    if not os.path.exists(path_gensim):
        st.warning(f"File model Word2Vec (Gensim) tidak ditemukan: {path_gensim}.")
        return None
    try:
        return Word2Vec.load(path_gensim)
    except Exception as e_gensim:
        st.error(f"Gagal memuat model Word2Vec (Gensim) '{path_gensim}': {e_gensim}")
        return None

# --- Inisialisasi Variabel dan Pemuatan Data Utama ---
predictions_df_app = load_csv_data_streamlit("dashboard_all_predictions.csv")
metrics_df_app = load_csv_data_streamlit("dashboard_metrics_summary.csv")
error_analysis_dfs_app = load_excel_data_streamlit('dashboard_error_analysis.xlsx')
label_encoder_app = load_pickle_object_streamlit("label_encoder.pkl")

if predictions_df_app.empty or metrics_df_app.empty or label_encoder_app is None:
    st.error(f"Gagal memuat data penting dari '{BASE_OUTPUT_PATH_FOR_APP}'. Pastikan Tahap 2 (persiapan data) berjalan sukses.")
    st.stop()

label_mapping_app = {i_map: label_encoder_app.classes_[i_map] for i_map in range(len(label_encoder_app.classes_))}
clickbait_label_text_app = next((cls_text_app for cls_text_app in label_encoder_app.classes_ if 'clickbait' in cls_text_app.lower()), None)
if clickbait_label_text_app:
    CLICKBAIT_CLASS_INDEX_APP = label_encoder_app.transform([clickbait_label_text_app])[0]
else:
    st.warning("Label 'clickbait' tidak terdeteksi. Default ke indeks 1 (atau 0).")
    CLICKBAIT_CLASS_INDEX_APP = 1 if len(label_encoder_app.classes_) > 1 else 0

if not metrics_df_app.empty:
    available_model_keys_app = metrics_df_app['Model_Key'].unique().tolist()
    available_model_display_names_app = metrics_df_app['Model_Display_Name'].unique().tolist()
else:
    available_model_keys_app = []
    available_model_display_names_app = ["Data model tidak tersedia"]

# --- Sidebar Navigasi ---
# st.sidebar.image("URL_ANDA_JIKA_ADA.png", width=60, caption="Nama Proyek") # Logo dihapus
st.sidebar.markdown("## 🧭 Menu Navigasi")
navigation_options_app = {
    "🚀 Ringkasan Kinerja": "Ringkasan Kinerja",
    "🔎 Detail Model": "Detail Model & Confusion Matrix",
    "🧐 Analisis Kesalahan": "Analisis Kesalahan",
    # "🔬 Analisis Fitur": "Analisis Fitur", # Dihapus dari navigasi
    "🔮 Prediksi Langsung": "Prediksi Langsung",
    "📄 Analisis File CSV": "Analisis File CSV"
}
selected_page_key_app = st.sidebar.radio(
    "Pilih Halaman:", list(navigation_options_app.keys()), key="sidebar_nav_app_v4" # Key diubah untuk update
)
selected_page_app = navigation_options_app[selected_page_key_app]

# --- Konten Halaman ---

if selected_page_app == "Ringkasan Kinerja":
    st.markdown("<h1 id='ringkasan-kinerja'>🚀 Ringkasan Kinerja Model</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.header("🚀 Ringkasan Performa Model")
    st.markdown("Analisis komparatif dari berbagai model klasifikasi yang diimplementasikan.")
    if not metrics_df_app.empty:
        st.subheader("Tabel Perbandingan Metrik")
        display_metrics_app = metrics_df_app.set_index('Model_Display_Name')[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
        st.dataframe(display_metrics_app.style.format("{:.4f}"))
        metric_to_plot_app = st.selectbox("Pilih Metrik untuk Grafik:", ['Accuracy', 'Precision', 'Recall', 'F1-Score'], key="ringkasan_metric_select_app")
        if metric_to_plot_app:
            fig_metrics_comp_app = px.bar(metrics_df_app, x='Model_Display_Name', y=metric_to_plot_app,
                                        color='Model_Display_Name', title=f"Grafik Perbandingan {metric_to_plot_app}",
                                        text_auto='.4f', labels={'Model_Display_Name': 'Model', metric_to_plot_app: metric_to_plot_app},
                                        color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_metrics_comp_app.update_layout(xaxis_title="Model & Representasi Fitur", yaxis_title=metric_to_plot_app,
                                           showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#E0E0E0'))
            st.plotly_chart(fig_metrics_comp_app, use_container_width=True)
    else: st.warning("Data metrik tidak tersedia.")
    st.subheader("Distribusi Label Aktual pada Data Uji")
    if 'true_label_text' in predictions_df_app.columns and not predictions_df_app.empty:
        true_label_counts_app = predictions_df_app['true_label_text'].value_counts()
        fig_dist_app = px.pie(values=true_label_counts_app.values, names=true_label_counts_app.index,
                          title="Distribusi Label Aktual", hole=0.4, color_discrete_sequence=px.colors.sequential.Plasma_r)
        fig_dist_app.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#E0E0E0'), legend_title_text='Kategori')
        st.plotly_chart(fig_dist_app, use_container_width=True)
    else: st.warning("Data label aktual tidak tersedia.")


elif selected_page_app == "Detail Model & Confusion Matrix":
    st.markdown("<h1 id='detail-model-dan-confusion-matrix'>🔎 Detail Model & Confusion Matrix</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("Lihat metrik performa lebih detail dan matriks kebingungan untuk setiap model.")
    selected_model_display_app = st.selectbox("Pilih Model:", available_model_display_names_app, key="detail_model_select_app_styled_v2") # Key diubah
    if selected_model_display_app and not metrics_df_app.empty and selected_model_display_app != "Data model tidak tersedia":
        model_key_series_app = metrics_df_app[metrics_df_app['Model_Display_Name'] == selected_model_display_app]['Model_Key']
        if not model_key_series_app.empty:
            model_key_app = model_key_series_app.iloc[0]
            model_metrics_series_app = metrics_df_app[metrics_df_app['Model_Key'] == model_key_app]
            if not model_metrics_series_app.empty:
                model_metrics_app = model_metrics_series_app.iloc[0]
                st.subheader(f"Metrik untuk: {selected_model_display_app}")
                m_col1_app, m_col2_app, m_col3_app, m_col4_app = st.columns(4)
                m_col1_app.metric("Accuracy", f"{model_metrics_app['Accuracy']:.4f}")
                m_col2_app.metric("Precision", f"{model_metrics_app['Precision']:.4f}")
                m_col3_app.metric("Recall", f"{model_metrics_app['Recall']:.4f}")
                m_col4_app.metric("F1-Score", f"{model_metrics_app['F1-Score']:.4f}")
                pred_col_num_app = f"pred_numeric_{model_key_app}"
                if pred_col_num_app in predictions_df_app.columns and 'true_label_numeric' in predictions_df_app.columns:
                    y_true_app = predictions_df_app['true_label_numeric']
                    y_pred_app = predictions_df_app[pred_col_num_app]
                    unique_labels_data_app = np.unique(np.concatenate((y_true_app, y_pred_app)))
                    cm_labels_text_app = [label_mapping_app.get(i_cm, str(i_cm)) for i_cm in sorted(unique_labels_data_app)]
                    cm_app = confusion_matrix(y_true_app, y_pred_app, labels=sorted(unique_labels_data_app))
                    fig_cm_app = px.imshow(cm_app, labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                                       x=cm_labels_text_app, y=cm_labels_text_app, text_auto=True,
                                       title=f"Confusion Matrix untuk {selected_model_display_app}",
                                       color_continuous_scale=px.colors.sequential.GnBu)
                    fig_cm_app.update_layout(xaxis_title="Label Prediksi", yaxis_title="Label Aktual",
                                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#E0E0E0'))
                    st.plotly_chart(fig_cm_app, use_container_width=True)
                else: st.warning(f"Kolom prediksi ('{pred_col_num_app}') atau label asli tidak ditemukan.")
            else: st.warning(f"Metrik untuk model '{selected_model_display_app}' tidak ditemukan.")
        else: st.warning(f"Kunci model untuk '{selected_model_display_app}' tidak ditemukan.")

elif selected_page_app == "Analisis Kesalahan":
    st.markdown("<h1 id='analisis-kesalahan-prediksi'>🧐 Analisis Kesalahan Prediksi</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("Tinjau contoh-contoh di mana model salah melakukan prediksi.")
    selected_model_display_err_app = st.selectbox("Pilih Model:", available_model_display_names_app, key="error_model_select_app_styled_v2") # Key diubah
    if selected_model_display_err_app and error_analysis_dfs_app and selected_model_display_err_app != "Data model tidak tersedia":
        model_key_err_series_app = metrics_df_app[metrics_df_app['Model_Display_Name'] == selected_model_display_err_app]['Model_Key']
        if not model_key_err_series_app.empty:
            model_key_err_app = model_key_err_series_app.iloc[0]
            if model_key_err_app in error_analysis_dfs_app:
                st.subheader(f"Contoh Prediksi Salah oleh: {selected_model_display_err_app}")
                df_errors_app = error_analysis_dfs_app[model_key_err_app]
                if not df_errors_app.empty:
                    st.dataframe(df_errors_app.style.set_properties(**{'white-space': 'pre-wrap', 'text-align': 'left'}), height=400, use_container_width=True)
                else: st.success(f"🎉 Tidak ada kesalahan prediksi yang tercatat untuk model {selected_model_display_err_app}!")
            else: st.warning(f"Data analisis kesalahan untuk '{model_key_err_app}' tidak ditemukan.")
        else: st.warning(f"Kunci model untuk '{selected_model_display_err_app}' tidak ditemukan.")

# Bagian Analisis Fitur DIHAPUS Sesuai Permintaan
# elif selected_page_app == "Analisis Fitur":
#     st.markdown("<h1 id='analisis-fitur-feature-importance'>🔬 Analisis Fitur</h1>", unsafe_allow_html=True)
#     st.markdown("---")
#     # ... (Kode Analisis Fitur sebelumnya) ...

elif selected_page_app == "Prediksi Langsung":
    st.markdown("<h1 id='prediksi-berita-baru'>🔮 Prediksi Berita Baru</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("Uji coba model klasifikasi dengan memasukkan judul berita baru di bawah ini.")
    selected_model_display_pred_app = st.selectbox("Pilih Model untuk Prediksi:", available_model_display_names_app, key="live_pred_model_select_app_styled_v2", help="Model yang dipilih akan digunakan untuk prediksi.") # Key diubah
    news_input_app = st.text_area("Masukkan Judul Berita:", height=150, key="live_pred_input_app_styled_v2", placeholder="Contoh: HEBOH! Penemuan Spesies Baru di Dasar Laut Terdalam Mengejutkan Ilmuwan!") # Key diubah
    if st.button("✨ Prediksi Sekarang!", key="live_pred_button_app_styled_v2"): # Key diubah
        if not news_input_app.strip(): st.warning("❗ Harap masukkan judul berita terlebih dahulu.")
        elif not selected_model_display_pred_app or selected_model_display_pred_app == "Data model tidak tersedia": st.warning("❗ Harap pilih model yang valid.")
        else:
            model_key_pred_series_app = metrics_df_app[metrics_df_app['Model_Display_Name'] == selected_model_display_pred_app]['Model_Key']
            if not model_key_pred_series_app.empty:
                model_key_pred_app = model_key_pred_series_app.iloc[0]
                model_pkl_pred_app = f"{model_key_pred_app.lower()}.pkl"
                model_to_use_app = load_pickle_object_streamlit(model_pkl_pred_app)
                if not model_to_use_app: st.error(f"❌ Gagal memuat model classifier '{model_pkl_pred_app}'.")
                else:
                    transformed_input_app = None
                    if "Original" in model_key_pred_app:
                        transformed_input_app = extract_basic_features_live_app([news_input_app])
                    elif "BoW" in model_key_pred_app:
                        vectorizer_app = load_pickle_object_streamlit("bow_vectorizer.pkl")
                        if vectorizer_app: transformed_input_app = vectorizer_app.transform([news_input_app])
                        else: st.error("❌ Gagal memuat BoW Vectorizer.")
                    elif "TFIDF" in model_key_pred_app:
                        vectorizer_app = load_pickle_object_streamlit("tfidf_vectorizer.pkl")
                        if vectorizer_app: transformed_input_app = vectorizer_app.transform([news_input_app])
                        else: st.error("❌ Gagal memuat TF-IDF Vectorizer.")
                    elif "W2V" in model_key_pred_app:
                        w2v_g_model_app = load_gensim_w2v_model_streamlit()
                        if w2v_g_model_app:
                            tokens_app = word_tokenize(news_input_app.lower())
                            transformed_input_app = np.array([get_w2v_embedding_live_app(tokens_app, w2v_g_model_app, W2V_VECTOR_SIZE_DASH_APP)])
                            transformed_input_app = np.nan_to_num(transformed_input_app)
                        else:
                            st.error("❌ Model Word2Vec (word2vec.model) untuk transformasi input gagal dimuat.")
                            transformed_input_app = None
                    if transformed_input_app is not None:
                        prediction_numeric_app = model_to_use_app.predict(transformed_input_app)[0]
                        prediction_text_app = label_mapping_app.get(prediction_numeric_app, f"Label_Unknown ({prediction_numeric_app})")
                        st.markdown("---")
                        col_hasil1_app, col_hasil2_app = st.columns([1,2])
                        with col_hasil1_app:
                            st.markdown("#### Hasil Klasifikasi:")
                            if prediction_numeric_app == CLICKBAIT_CLASS_INDEX_APP: st.error(f"**{prediction_text_app}**")
                            else: st.success(f"**{prediction_text_app}**")
                        if hasattr(model_to_use_app, "predict_proba"):
                            with col_hasil2_app:
                                prediction_proba_app = model_to_use_app.predict_proba(transformed_input_app)[0]
                                st.markdown("#### Skor Kepercayaan:")
                                for i_app_proba in range(len(label_encoder_app.classes_)):
                                    class_label_text_proba_app = label_encoder_app.classes_[i_app_proba]
                                    prob_value_app = prediction_proba_app[i_app_proba]
                                    st.write(f"{class_label_text_proba_app}:")
                                    st.progress(float(prob_value_app), text=f"{prob_value_app*100:.1f}%")
                    elif "W2V" in model_key_pred_app and not w2v_g_model_app: pass
                    else: st.error("❌ Gagal mentransformasi input. Periksa apakah vectorizer/model Word2Vec (Gensim) termuat.")
            else: st.warning(f"❗ Tidak dapat menemukan kunci model untuk '{selected_model_display_pred_app}'.")

elif selected_page_app == "Analisis File CSV":
    st.markdown("<h1 id='analisis-file-csv-unggahan'>📄 Analisis File CSV Unggahan</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("Unggah file CSV Anda yang berisi judul berita untuk diklasifikasikan. Pastikan file CSV memiliki kolom yang berisi teks judul berita.")

    uploaded_file_app = st.file_uploader("Pilih file CSV (maks. 200MB)", type=["csv"], key="csv_uploader_app_v2", help="File CSV dengan satu kolom berisi teks judul berita.") # Key diubah

    if uploaded_file_app is not None:
        try:
            df_uploaded_app = pd.read_csv(uploaded_file_app)
            st.success("✔️ File CSV berhasil diunggah!")
            st.subheader("Pratinjau Data Unggahan (5 baris pertama):")
            st.dataframe(df_uploaded_app.head())

            text_column_options_app = [""] + df_uploaded_app.columns.tolist()
            selected_text_column_app = st.selectbox(
                "Pilih kolom yang berisi teks/judul berita:",
                text_column_options_app,
                index=0,
                key="csv_text_column_select_app_v2", # Key diubah
                help="Kolom ini akan digunakan sebagai input untuk analisis."
            )

            # Menggunakan nama variabel yang konsisten: selected_model_display_csv_app
            selected_model_display_csv_app = st.selectbox(
                "Pilih model untuk prediksi pada file ini:",
                available_model_display_names_app,
                key="csv_analysis_model_select_app_v2", # Key diubah
                help="Model yang akan digunakan untuk mengklasifikasikan teks dari file CSV."
            )

            if st.button("🚀 Proses dan Prediksi CSV!", key="process_csv_button_app_v2") and                selected_text_column_app and                selected_model_display_csv_app and                selected_model_display_csv_app != "Data model tidak tersedia": # Menggunakan variabel yang benar

                with st.spinner("⏳ Memproses data dan melakukan prediksi... Mohon tunggu."):
                    texts_to_analyze_app = df_uploaded_app[selected_text_column_app].astype(str).fillna('')

                    # Menggunakan selected_model_display_csv_app untuk mendapatkan model_key
                    model_key_analysis_series_app = metrics_df_app[metrics_df_app['Model_Display_Name'] == selected_model_display_csv_app]['Model_Key']

                    if not model_key_analysis_series_app.empty:
                        model_key_analysis_app = model_key_analysis_series_app.iloc[0]
                        model_pkl_analysis_app = f"{model_key_analysis_app.lower()}.pkl"
                        classifier_model_app = load_pickle_object_streamlit(model_pkl_analysis_app)

                        if not classifier_model_app:
                            st.error(f"❌ Gagal memuat model classifier '{model_pkl_analysis_app}'. Analisis dibatalkan.")
                        else:
                            features_for_prediction_app = None
                            if "Original" in model_key_analysis_app:
                                features_for_prediction_app = extract_basic_features_live_app(texts_to_analyze_app)
                            elif "BoW" in model_key_analysis_app:
                                bow_vectorizer_app = load_pickle_object_streamlit("bow_vectorizer.pkl")
                                if bow_vectorizer_app: features_for_prediction_app = bow_vectorizer_app.transform(texts_to_analyze_app)
                                else: st.error("❌ Gagal memuat BoW Vectorizer.")
                            elif "TFIDF" in model_key_analysis_app:
                                tfidf_vectorizer_app = load_pickle_object_streamlit("tfidf_vectorizer.pkl")
                                if tfidf_vectorizer_app: features_for_prediction_app = tfidf_vectorizer_app.transform(texts_to_analyze_app)
                                else: st.error("❌ Gagal memuat TF-IDF Vectorizer.")
                            elif "W2V" in model_key_analysis_app:
                                w2v_g_model_live_app = load_gensim_w2v_model_streamlit()
                                if w2v_g_model_live_app:
                                    tokenized_texts_live_app = [word_tokenize(text_app.lower()) for text_app in texts_to_analyze_app]
                                    embeddings_live_app = [get_w2v_embedding_live_app(tokens_app, w2v_g_model_live_app, W2V_VECTOR_SIZE_DASH_APP) for tokens_app in tokenized_texts_live_app]
                                    features_for_prediction_app = np.array(embeddings_live_app)
                                    features_for_prediction_app = np.nan_to_num(features_for_prediction_app)
                                else:
                                    st.error("❌ Model Word2Vec (Gensim) gagal dimuat.")
                                    features_for_prediction_app = None

                            if features_for_prediction_app is not None:
                                predictions_numeric_csv_app = classifier_model_app.predict(features_for_prediction_app)
                                predictions_text_csv_app = label_encoder_app.inverse_transform(predictions_numeric_csv_app)
                                df_uploaded_app[f'Prediksi_{model_key_analysis_app}'] = predictions_text_csv_app
                                if hasattr(classifier_model_app, "predict_proba"):
                                    try:
                                        probas_csv_app = classifier_model_app.predict_proba(features_for_prediction_app)
                                        df_uploaded_app[f'Prob_Clickbait_{model_key_analysis_app}'] = probas_csv_app[:, CLICKBAIT_CLASS_INDEX_APP]
                                    except Exception as e_proba_csv:
                                        st.warning(f"Tidak dapat menghitung probabilitas untuk model {model_key_analysis_app}: {e_proba_csv}")
                                st.subheader("✔️ Hasil Prediksi pada Data Unggahan:")
                                st.dataframe(df_uploaded_app)
                                @st.cache_data
                                def convert_df_to_csv_app(df_to_convert_app):
                                    return df_to_convert_app.to_csv(index=False).encode('utf-8')
                                csv_output_app = convert_df_to_csv_app(df_uploaded_app)
                                st.download_button(
                                    label="📥 Unduh Hasil Prediksi (CSV)", data=csv_output_app,
                                    file_name=f"hasil_prediksi_{uploaded_file_app.name}", mime="text/csv",
                                    key="download_csv_pred_app_v2" # Key diubah
                                )
                                st.subheader("Ringkasan Hasil Prediksi:")
                                prediction_counts_csv_app = df_uploaded_app[f'Prediksi_{model_key_analysis_app}'].value_counts()
                                fig_pred_dist_csv_app = px.pie(values=prediction_counts_csv_app.values,
                                                       names=prediction_counts_csv_app.index,
                                                       title="Distribusi Hasil Prediksi dari File CSV", hole=0.3,
                                                       color_discrete_sequence=px.colors.sequential.Aggrnyl)
                                fig_pred_dist_csv_app.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#E0E0E0'))
                                st.plotly_chart(fig_pred_dist_csv_app, use_container_width=True)
                            else:
                                st.error("❌ Gagal mengekstrak fitur dari data unggahan. Analisis dibatalkan.")
                    else:
                        st.warning("❗ Kunci model tidak ditemukan untuk model yang dipilih.")

        except pd.errors.EmptyDataError:
            st.error("❌ File CSV yang diunggah kosong atau tidak valid.")
        except Exception as e_csv_process:
            st.error(f"Terjadi kesalahan saat memproses file CSV: {e_csv_process}")
            st.error("Pastikan format file CSV benar dan kolom teks yang dipilih valid.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #8B949E; font-size: 0.9em;'>Dashboard Analytica Clickbait © 2024</p>", unsafe_allow_html=True)
