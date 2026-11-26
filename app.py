import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, silhouette_score
)

st.set_page_config(
    page_title="ML App — TAS Mata Kuliah AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "custom_df" not in st.session_state:
    st.session_state.custom_df = None
if "komentar_list" not in st.session_state:
    st.session_state.komentar_list = [
        {"Nama": "Mahasiswa AI", "Pesan": "Aplikasi demo ML sangat membantu memahami perbandingan model!", "Waktu": "2026-11-26"}
    ]

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan Model")
    st.write("Atur parameter di bawah, lalu klik **Train Model** untuk memulai training.")

    task = st.radio(
        "🎯 Pilih Jenis Task",
        ["Regresi", "Klasifikasi", "Clustering", "Manajemen Dataset", "Komentar Publik"]
    )

    if task in ["Regresi", "Klasifikasi", "Clustering"]:
        sumber_data = st.radio("📁 Sumber Data", ["Scikit-learn", "MongoDB"])

        if sumber_data == "Scikit-learn":
            if task == "Regresi":
                dataset_name = st.selectbox("📊 Pilih Dataset", ["Diabetes", "California Housing", "Linnerud"])
            else:
                dataset_name = st.selectbox("📊 Pilih Dataset", ["Iris", "Wine", "Breast Cancer"])
        else:
            dataset_name = "Dataset MongoDB / CSV"

        if task == "Regresi":
            model_name = st.selectbox(
                "🧠 Pilih Model",
                ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree Regressor", "Random Forest Regressor"]
            )
        elif task == "Klasifikasi":
            model_name = st.selectbox(
                "🧠 Pilih Model",
                ["Logistic Regression", "K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)", "Decision Tree Classifier", "Random Forest Classifier"]
            )
        elif task == "Clustering":
            model_name = st.selectbox(
                "🧠 Pilih Model",
                ["K-Means Clustering", "Agglomerative Clustering", "DBSCAN"]
            )
            n_clusters = st.slider("🔢 Jumlah Cluster (K)", 2, 8, 3)

        if task in ["Regresi", "Klasifikasi"]:
            test_size = st.slider(
                "📐 Test Size",
                min_value=0.10, max_value=0.50, value=0.20, step=0.05,
                help="Proporsi data yang digunakan untuk pengujian (testing)."
            )
            random_state = st.number_input("🎲 Random State", min_value=0, max_value=999, value=42, step=1)

        train_btn = st.button("🚀 Train Model", type="primary", use_container_width=True)

        st.divider()
        st.info("💡 **Tips:** Coba berbagai kombinasi dataset & model!")


def get_dataset(task_type, source, name):
    if source == "MongoDB":
        if st.session_state.custom_df is None:
            return None, None, None
        df = st.session_state.custom_df.select_dtypes(include=[np.number]).dropna()
        if df.shape[1] < 2:
            return None, None, None
        return df.iloc[:, :-1], df.iloc[:, -1], df

    if task_type == "Regresi":
        if name == "Diabetes":
            data = datasets.load_diabetes(as_frame=True)
        elif name == "California Housing":
            data = datasets.fetch_california_housing(as_frame=True)
        else:
            data = datasets.load_linnerud(as_frame=True)
            df = pd.concat([data.data, data.target.iloc[:, 0]], axis=1)
            return data.data, data.target.iloc[:, 0], df
    else:
        if name == "Iris":
            data = datasets.load_iris(as_frame=True)
        elif name == "Wine":
            data = datasets.load_wine(as_frame=True)
        else:
            data = datasets.load_breast_cancer(as_frame=True)

    return data.data, data.target, data.frame


# ==================== HALAMAN UTAMA ====================
if task in ["Regresi", "Klasifikasi", "Clustering"]:
    icon_map = {"Regresi": "📈", "Klasifikasi": "🏷️", "Clustering": "🔮"}
    st.title(f"{icon_map[task]} {task} Machine Learning")
    st.write("Aplikasi demo Machine Learning menggunakan dataset dari scikit-learn atau MongoDB.")

    if not train_btn:
        st.info("👈 Silakan atur parameter di sidebar, lalu klik 🚀 **Train Model** untuk memulai training.")
        st.write("")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📊 Dataset")
            st.caption("Scikit-learn atau MongoDB")
        with col2:
            st.markdown("### 🧠 Model")
            st.caption("5 model regresi tersedia")
        with col3:
            st.markdown("### 📐 Evaluasi")
            st.caption("Metrics lengkap + visualisasi")
    else:
        X, y, df_full = get_dataset(task, sumber_data, dataset_name)

        if X is None:
            st.warning("⚠️ Dataset belum tersedia atau kolom numerik kurang dari 2. Silakan unggah CSV di menu **Manajemen Dataset**.")
        else:
            with st.expander("🔍 Lihat Preview Dataset", expanded=False):
                st.write(f"**Dimensi Data:** {df_full.shape[0]} baris × {df_full.shape[1]} kolom")
                st.dataframe(df_full.head(10), use_container_width=True)

            if task == "Regresi":
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
                models = {
                    "Linear Regression": LinearRegression(),
                    "Ridge Regression": Ridge(random_state=int(random_state)),
                    "Lasso Regression": Lasso(random_state=int(random_state)),
                    "Decision Tree Regressor": DecisionTreeRegressor(random_state=int(random_state)),
                    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=int(random_state))
                }
                model = models[model_name]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)

                st.success(f"✅ Training selesai menggunakan **{model_name}** pada dataset **{dataset_name}**!")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R² Score", f"{r2:.4f}")
                m2.metric("MAE", f"{mae:.4f}")
                m3.metric("MSE", f"{mse:.4f}")
                m4.metric("RMSE", f"{rmse:.4f}")

                st.subheader("📊 Visualisasi Aktual vs Prediksi")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(y_test, y_pred, alpha=0.7, color="#ff4b4b", edgecolors="k")
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "w--", lw=2)
                ax.set_xlabel("Nilai Aktual")
                ax.set_ylabel("Nilai Prediksi")
                ax.set_title(f"Actual vs Predicted ({model_name})")
                fig.patch.set_facecolor("#0e1117")
                ax.set_facecolor("#262730")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.title.set_color("white")
                st.pyplot(fig)

            elif task == "Klasifikasi":
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
                scaler = StandardScaler()
                X_train_sc = scaler.fit_transform(X_train)
                X_test_sc = scaler.transform(X_test)

                models = {
                    "Logistic Regression": LogisticRegression(max_iter=500),
                    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
                    "Support Vector Machine (SVM)": SVC(),
                    "Decision Tree Classifier": DecisionTreeClassifier(random_state=int(random_state)),
                    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=int(random_state))
                }
                model = models[model_name]
                model.fit(X_train_sc, y_train)
                y_pred = model.predict(X_test_sc)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                st.success(f"✅ Training selesai menggunakan **{model_name}** pada dataset **{dataset_name}**!")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{acc*100:.2f}%")
                m2.metric("Precision", f"{prec:.4f}")
                m3.metric("Recall", f"{rec:.4f}")
                m4.metric("F1-Score", f"{f1:.4f}")

                st.subheader("📊 Confusion Matrix")
                fig, ax = plt.subplots(figsize=(6, 4))
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(ax=ax, cmap="Reds", colorbar=False)
                fig.patch.set_facecolor("#0e1117")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                st.pyplot(fig)

            elif task == "Clustering":
                scaler = StandardScaler()
                X_sc = scaler.fit_transform(X)

                if model_name == "K-Means Clustering":
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                elif model_name == "Agglomerative Clustering":
                    model = AgglomerativeClustering(n_clusters=n_clusters)
                else:
                    model = DBSCAN(eps=0.8, min_samples=5)

                labels = model.fit_predict(X_sc)
                unique_labels = len(set(labels)) - (1 if -1 in labels else 0)

                st.success(f"✅ Clustering selesai menggunakan **{model_name}**!")
                c1, c2 = st.columns(2)
                c1.metric("Jumlah Cluster Terbentuk", unique_labels)
                if unique_labels > 1:
                    sil = silhouette_score(X_sc, labels)
                    c2.metric("Silhouette Score", f"{sil:.4f}")
                else:
                    c2.metric("Silhouette Score", "N/A")

                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_sc)
                fig, ax = plt.subplots(figsize=(8, 4))
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.8, edgecolors="k")
                ax.set_xlabel("PCA Component 1")
                ax.set_ylabel("PCA Component 2")
                ax.set_title(f"Visualisasi Cluster 2D ({model_name})")
                fig.patch.set_facecolor("#0e1117")
                ax.set_facecolor("#262730")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.title.set_color("white")
                plt.colorbar(scatter, ax=ax)
                st.pyplot(fig)

elif task == "Manajemen Dataset":
    st.title("📁 Manajemen Dataset")
    st.write("Unggah file dataset berformat **CSV** untuk digunakan pada menu Regresi, Klasifikasi, atau Clustering (pilih sumber data **MongoDB**).")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    if uploaded_file is not None:
        df_up = pd.read_csv(uploaded_file)
        st.session_state.custom_df = df_up
        st.success(f"Dataset **{uploaded_file.name}** berhasil dimuat ({df_up.shape[0]} baris, {df_up.shape[1]} kolom).")

    if st.session_state.custom_df is not None:
        st.subheader("Preview Dataset Tersimpan")
        st.dataframe(st.session_state.custom_df, use_container_width=True)
        if st.button("🗑️ Hapus Dataset Kustom"):
            st.session_state.custom_df = None
            st.rerun()

elif task == "Komentar Publik":
    st.title("💬 Komentar Publik")
    st.write("Tinggalkan masukan, ulasan, atau catatan pengujian aplikasi di bawah ini.")

    with st.form("form_komentar", clear_on_submit=True):
        nama = st.text_input("Nama Anda", placeholder="Contoh: Budi Santoso")
        pesan = st.text_area("Komentar / Masukan", placeholder="Tulis komentar Anda di sini...")
        kirim = st.form_submit_button("Kirim Komentar", type="primary")
        if kirim and nama.strip() and pesan.strip():
            st.session_state.komentar_list.insert(0, {
                "Nama": nama.strip(),
                "Pesan": pesan.strip(),
                "Waktu": "Baru saja"
            })
            st.success("Komentar berhasil ditambahkan!")

    st.subheader("Daftar Komentar")
    for item in st.session_state.komentar_list:
        with st.container(border=True):
            st.markdown(f"**{item['Nama']}** · *{item['Waktu']}*")
            st.write(item["Pesan"])

st.divider()
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.9em;'>"
    "© TAS Mata Kuliah AI — UKSW 2026"
    "</div>",
    unsafe_allow_html=True
)

st.divider()
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.9em;'>"
    "© 2026 — Dikembangkan oleh <b>Al Fira Anugrah</b> | TAS Mata Kuliah AI"
    "</div>",
    unsafe_allow_html=True
)