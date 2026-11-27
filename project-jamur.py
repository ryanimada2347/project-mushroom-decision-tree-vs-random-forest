import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

st.set_page_config(
    page_title="Mushroom AI | Classification Dashboard",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9CA3AF;
        font-size: 1rem;
        margin-top: 0.2rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 15px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">🍄 Mushroom Classification</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Dashboard analisis dan perbandingan algoritma Decision Tree & Random Forest</p>',
    unsafe_allow_html=True,
)
st.divider()

with st.sidebar:
    st.title("🍄 Mushroom AI")
    st.caption("Classification Dashboard")
    page = st.radio(
        "Navigasi",
        [
            "🏠 Dashboard",
            "📊 Exploratory Data Analysis",
            "⚙️ Preprocessing",
            "🤖 Perbandingan Model",
            "🌳 Visualisasi Decision Tree",
            "🔎 Prediksi Jamur",
            "📁 Dataset",
        ],
    )
    st.divider()
    st.subheader("Pengaturan Model")
    test_size = st.slider("Persentase data testing", 10, 40, 20, 5)
    random_state = st.number_input("Random state", 0, 9999, 1)
    dt_depth = st.slider("Kedalaman Decision Tree", 1, 15, 3)
    rf_estimators = st.slider("Jumlah pohon Random Forest", 50, 500, 100, 50)
    rf_depth_option = st.selectbox(
        "Kedalaman Random Forest",
        ["Tidak dibatasi", "5", "10", "15", "20"],
    )
    rf_depth = None if rf_depth_option == "Tidak dibatasi" else int(rf_depth_option)
    st.caption("Perubahan pengaturan akan melatih ulang model.")

@st.cache_data
def load_local_dataset():
    file_path = Path(__file__).parent / "mushrooms.csv"
    return pd.read_csv(file_path)

try:
    dataset = load_local_dataset()
except FileNotFoundError:
    st.error("File mushrooms.csv tidak ditemukan. Letakkan file CSV di folder yang sama dengan project-jamur.py.")
    st.stop()

if "class" not in dataset.columns:
    st.error("Kolom target 'class' tidak ditemukan pada dataset.")
    st.stop()

if dataset.empty:
    st.error("Dataset kosong.")
    st.stop()

data = dataset.copy()
data = data.replace("?", np.nan)
data = data.fillna("Unknown")
data["class"] = data["class"].astype(str)

@st.cache_resource
def train_models(dataframe, test_fraction, seed, tree_depth, forest_trees, forest_depth):
    X_raw = dataframe.drop(columns=["class"])
    y_raw = dataframe["class"]

    X_encoded = pd.get_dummies(X_raw, dtype=int)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y_encoded,
        test_size=test_fraction,
        random_state=seed,
        stratify=y_encoded,
    )

    decision_tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=tree_depth,
        random_state=seed,
    )
    random_forest = RandomForestClassifier(
        n_estimators=forest_trees,
        criterion="gini",
        max_depth=forest_depth,
        random_state=seed,
        n_jobs=-1,
    )

    decision_tree.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)

    return (
        X_raw, X_encoded, label_encoder, X_train, X_test, y_train, y_test,
        decision_tree, random_forest,
    )

try:
    (
        X_raw, X, encoder, X_train, X_test, y_train, y_test,
        dt_model, rf_model,
    ) = train_models(
        data,
        test_size / 100,
        int(random_state),
        dt_depth,
        rf_estimators,
        rf_depth,
    )
except ValueError as error:
    st.error(f"Model tidak dapat dilatih: {error}")
    st.stop()

class_names = [
    "Edible (Dapat dimakan)" if str(value).lower() == "e"
    else "Poisonous (Beracun)" if str(value).lower() == "p"
    else str(value)
    for value in encoder.classes_
]

y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

def calculate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

dt_metrics = calculate_metrics(y_test, y_pred_dt)
rf_metrics = calculate_metrics(y_test, y_pred_rf)

comparison = pd.DataFrame(
    [dt_metrics, rf_metrics],
    index=["Decision Tree", "Random Forest"],
)

if dt_metrics["Accuracy"] > rf_metrics["Accuracy"]:
    best_model_name = "Decision Tree"
elif rf_metrics["Accuracy"] > dt_metrics["Accuracy"]:
    best_model_name = "Random Forest"
else:
    best_model_name = "Keduanya memiliki akurasi sama"

if page == "🏠 Dashboard":
    st.header("Ringkasan Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data", f"{len(data):,}")
    c2.metric("Jumlah Fitur", len(data.columns) - 1)
    c3.metric("Jumlah Kelas", data["class"].nunique())
    c4.metric("Data Training", f"{len(X_train):,}")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Distribusi Kelas Jamur")
        class_counts = data["class"].value_counts().rename(
            index={"e": "Edible", "p": "Poisonous"}
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax)
        ax.set_xlabel("Kategori")
        ax.set_ylabel("Jumlah data")
        ax.set_title("Jumlah Jamur Berdasarkan Kelas")
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.subheader("Performa Model")
        metrics_chart = comparison[
            ["Accuracy", "Precision", "Recall", "F1-Score"]
        ].T
        fig, ax = plt.subplots(figsize=(7, 4))
        metrics_chart.plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Skor")
        ax.set_xlabel("Metrik")
        ax.set_title("Decision Tree vs Random Forest")
        ax.legend(title="Model")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Hasil Evaluasi Singkat")
    m1, m2 = st.columns(2)
    m1.metric("Decision Tree Accuracy", f"{dt_metrics['Accuracy'] * 100:.2f}%")
    m2.metric("Random Forest Accuracy", f"{rf_metrics['Accuracy'] * 100:.2f}%")

    st.success(
        f"Model dengan accuracy tertinggi pada data testing: {best_model_name}."
    )
    st.info(
        "Catatan: hasil ini berasal dari pembagian data training/testing dan pengaturan model "
        "yang sedang dipilih pada sidebar."
    )

elif page == "📊 Exploratory Data Analysis":
    st.header("Exploratory Data Analysis (EDA)")
    st.write(
        "Bagian ini digunakan untuk memahami isi, distribusi, dan karakteristik dataset."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Baris", f"{data.shape[0]:,}")
    c2.metric("Kolom", data.shape[1])
    c3.metric("Nilai kosong awal", int(dataset.isnull().sum().sum()))

    st.subheader("Distribusi kelas")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(
        data=data,
        x="class",
        order=dataset["class"].value_counts().index,
        ax=ax,
    )
    ax.set_xticklabels(
        [
            "Edible" if tick.get_text().lower() == "e"
            else "Poisonous" if tick.get_text().lower() == "p"
            else tick.get_text()
            for tick in ax.get_xticklabels()
        ]
    )
    ax.set_xlabel("Kelas jamur")
    ax.set_ylabel("Jumlah")
    ax.set_title("Distribusi Edible dan Poisonous")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Distribusi fitur kategorikal")
    selected_feature = st.selectbox("Pilih fitur", list(X_raw.columns))
    feature_counts = data[selected_feature].value_counts().head(20)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=feature_counts.values,
        y=feature_counts.index.astype(str),
        ax=ax,
    )
    ax.set_xlabel("Jumlah")
    ax.set_ylabel(selected_feature)
    ax.set_title(f"Distribusi {selected_feature} (maksimal 20 kategori)")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Hubungan fitur dengan kelas")
    relation_feature = st.selectbox(
        "Pilih fitur untuk dibandingkan dengan kelas",
        list(X_raw.columns),
        key="relation",
    )

    cross_tab = pd.crosstab(data[relation_feature], data["class"])
    cross_tab = cross_tab.rename(
        columns={"e": "Edible", "p": "Poisonous"}
    )
    st.dataframe(cross_tab, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    cross_tab.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(f"Perbandingan kelas berdasarkan {relation_feature}")
    ax.set_xlabel(relation_feature)
    ax.set_ylabel("Jumlah")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close(fig)

elif page == "⚙️ Preprocessing":
    st.header("Preprocessing Data")
    st.markdown(
        "Tahapan pengolahan data sebelum digunakan untuk melatih model."
    )

    st.subheader("1. Pemeriksaan data")
    st.write("Ukuran dataset:", dataset.shape)

    missing_table = pd.DataFrame({
        "Kolom": dataset.columns,
        "Missing Values": dataset.isnull().sum().values,
        "Tipe Data": dataset.dtypes.astype(str).values,
    })

    st.dataframe(missing_table, use_container_width=True)

    st.subheader("2. Mengubah data kategori menjadi numerik")
    st.write(
        "Metode yang digunakan: **One-Hot Encoding** pada fitur dan "
        "**Label Encoding** pada kolom class."
    )
    st.write(f"Jumlah fitur sebelum encoding: {X_raw.shape[1]}")
    st.write(f"Jumlah fitur setelah encoding: {X.shape[1]}")

    with st.expander("Lihat hasil One-Hot Encoding"):
        st.dataframe(X.head(10), use_container_width=True)

    st.subheader("3. Pembagian data")
    c1, c2 = st.columns(2)
    c1.metric("Data Training", len(X_train))
    c2.metric("Data Testing", len(X_test))

    st.write(f"Rasio training/testing: {100-test_size}% / {test_size}%")
    st.caption(
        "Stratify digunakan agar proporsi kelas pada data training dan testing tetap terjaga."
    )

elif page == "🤖 Perbandingan Model":
    st.header("Evaluasi dan Perbandingan Model")
    st.write("Kedua model diuji menggunakan data testing yang sama.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Decision Tree Accuracy",
        f"{dt_metrics['Accuracy']*100:.2f}%",
    )
    c2.metric(
        "Random Forest Accuracy",
        f"{rf_metrics['Accuracy']*100:.2f}%",
    )
    c3.metric(
        "Selisih Accuracy",
        f"{abs(dt_metrics['Accuracy']-rf_metrics['Accuracy'])*100:.2f}%",
    )
    c4.metric("Model Terbaik", best_model_name)

    st.subheader("Tabel perbandingan")
    st.dataframe((comparison * 100).round(2), use_container_width=True)

    st.subheader("Grafik perbandingan metrik")
    fig, ax = plt.subplots(figsize=(10, 5))
    comparison.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Skor")
    ax.set_xlabel("Algoritma")
    ax.set_title("Perbandingan Decision Tree dan Random Forest")
    ax.legend(title="Metrik")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")
    tab1, tab2 = st.tabs(["Decision Tree", "Random Forest"])

    for tab, predictions in [(tab1, y_pred_dt), (tab2, y_pred_rf)]:
        with tab:
            report = classification_report(
                y_test,
                predictions,
                labels=np.arange(len(class_names)),
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
            st.dataframe(
                pd.DataFrame(report).transpose().round(4),
                use_container_width=True,
            )

    st.download_button(
        "⬇️ Unduh tabel perbandingan (CSV)",
        data=(comparison * 100).round(2).to_csv().encode("utf-8"),
        file_name="perbandingan_decision_tree_random_forest.csv",
        mime="text/csv",
    )

elif page == "🌳 Visualisasi Decision Tree":
    st.header("Visualisasi Decision Tree")
    st.write(
        f"Model menggunakan criterion **Gini** dan max_depth **{dt_depth}**."
    )

    fig, ax = plt.subplots(figsize=(22, 11))
    plot_tree(
        dt_model,
        feature_names=X.columns,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=7,
        ax=ax,
    )
    ax.set_title("Decision Tree - Gini")
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.header("Confusion Matrix")

    tab1, tab2 = st.tabs(["Decision Tree", "Random Forest"])

    cm_dt = confusion_matrix(
        y_test,
        y_pred_dt,
        labels=np.arange(len(class_names)),
    )
    cm_rf = confusion_matrix(
        y_test,
        y_pred_rf,
        labels=np.arange(len(class_names)),
    )

    with tab1:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm_dt,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title("Confusion Matrix - Decision Tree")
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm_rf,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title("Confusion Matrix - Random Forest")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Feature Importance - Random Forest")

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature",
        ax=ax,
    )
    ax.set_title("20 Fitur Paling Berpengaruh")
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(importance, use_container_width=True)

elif page == "🔎 Prediksi Jamur":
    st.header("Prediksi Karakteristik Jamur")
    st.write(
        "Pilih karakteristik jamur pada setiap kolom. Model akan memberikan prediksi "
        "berdasarkan pola yang dipelajari dari dataset."
    )

    st.warning(
        "Prediksi ini hanya untuk demonstrasi pembelajaran machine learning. "
        "Jangan gunakan aplikasi ini untuk menentukan keamanan jamur liar yang akan dimakan."
    )

    with st.form("mushroom_prediction_form"):
        input_values = {}
        columns = st.columns(2)

        for index, feature in enumerate(X_raw.columns):
            options = sorted(X_raw[feature].astype(str).unique().tolist())

            with columns[index % 2]:
                input_values[feature] = st.selectbox(
                    feature.replace("-", " ").replace("_", " ").title(),
                    options,
                    key=f"predict_{feature}",
                )

        submitted = st.form_submit_button(
            "🔍 Prediksi dengan Kedua Model",
            use_container_width=True,
        )

    if submitted:
        input_df = pd.DataFrame([input_values])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(
            columns=X.columns,
            fill_value=0,
        )

        dt_prediction = int(dt_model.predict(input_encoded)[0])
        rf_prediction = int(rf_model.predict(input_encoded)[0])

        dt_label = class_names[dt_prediction]
        rf_label = class_names[rf_prediction]

        p1, p2 = st.columns(2)

        with p1:
            st.subheader("Decision Tree")
            st.metric("Hasil prediksi", dt_label)

            probabilities = dt_model.predict_proba(input_encoded)[0]
            st.write("Probabilitas prediksi:")
            st.progress(float(max(probabilities)))
            st.write(f"{max(probabilities)*100:.2f}%")

        with p2:
            st.subheader("Random Forest")
            st.metric("Hasil prediksi", rf_label)

            probabilities = rf_model.predict_proba(input_encoded)[0]
            st.write("Probabilitas prediksi:")
            st.progress(float(max(probabilities)))
            st.write(f"{max(probabilities)*100:.2f}%")

        if dt_prediction == rf_prediction:
            st.success(
                "Kedua model memberikan hasil prediksi yang sama."
            )
        else:
            st.info(
                "Kedua model memberikan hasil berbeda. Periksa kembali karakteristik input dan evaluasi model."
            )

elif page == "📁 Dataset":
    st.header("Dataset Mushroom Classification")
    st.write("Gunakan pencarian dan filter untuk menjelajahi data.")

    search_text = st.text_input("Cari nilai pada dataset")
    filtered_data = data.copy()

    if search_text:
        mask = filtered_data.astype(str).apply(
            lambda column: column.str.contains(
                search_text,
                case=False,
                na=False,
            )
        ).any(axis=1)

        filtered_data = filtered_data[mask]

    selected_class = st.multiselect(
        "Filter kelas",
        options=sorted(data["class"].unique()),
        default=sorted(data["class"].unique()),
        format_func=lambda value: {
            "e": "Edible (e)",
            "p": "Poisonous (p)",
        }.get(value.lower(), value),
    )

    filtered_data = filtered_data[
        filtered_data["class"].isin(selected_class)
    ]

    st.caption(
        f"Menampilkan {len(filtered_data):,} dari {len(data):,} baris."
    )

    st.dataframe(
        filtered_data,
        use_container_width=True,
        height=450,
    )

    st.download_button(
        "⬇️ Unduh dataset yang ditampilkan (CSV)",
        data=filtered_data.to_csv(index=False).encode("utf-8"),
        file_name="mushroom_dataset_filtered.csv",
        mime="text/csv",
    )

st.divider()

st.caption(
    "Mushroom Classification Dashboard | Decision Tree vs Random Forest | "
    "Dibuat menggunakan Python, Scikit-learn, Pandas, Seaborn, dan Streamlit."
)