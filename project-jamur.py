
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
    classification_report
)

st.set_page_config(
    page_title="Mushroom Classification",
    page_icon="🍄",
    layout="wide"
)

st.title("🍄 Mushroom Classification")
st.subheader("Perbandingan Decision Tree dan Random Forest")
st.write(
    "Aplikasi klasifikasi jamur menggunakan algoritma "
    "Decision Tree dan Random Forest."
)

@st.cache_data
def load_dataset():
    file_path = Path(__file__).parent / "mushrooms.csv"
    data = pd.read_csv(file_path)
    return data

try:
    dataset = load_dataset()
except FileNotFoundError:
    st.error("File mushrooms.csv tidak ditemukan!")
    st.stop()

if "class" not in dataset.columns:
    st.error("Kolom target 'class' tidak ditemukan!")
    st.stop()

st.header("1. Informasi Dataset")

col1, col2, col3 = st.columns(3)

col1.metric("Jumlah Data", dataset.shape[0])
col2.metric("Jumlah Fitur", dataset.shape[1] - 1)
col3.metric("Missing Values", int(dataset.isnull().sum().sum()))

with st.expander("Lihat Dataset"):
    st.dataframe(dataset.head(10), width="stretch")

st.subheader("Distribusi Kelas Jamur")

fig, ax = plt.subplots(figsize=(7, 4))
sns.countplot(data=dataset, x="class", ax=ax)
ax.set_title("Distribusi Edible dan Poisonous")
ax.set_xlabel("Class")
ax.set_ylabel("Jumlah")
st.pyplot(fig)
plt.close(fig)

st.header("2. Preprocessing Data")

X = dataset.drop(["class"], axis=1)
y = dataset["class"]

X = pd.get_dummies(X)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

class_names = [
    "Edible" if str(c).lower() == "e"
    else "Poisonous" if str(c).lower() == "p"
    else str(c)
    for c in encoder.classes_
]

st.write("Encoding fitur kategorikal berhasil.")
st.write("Jumlah fitur setelah One-Hot Encoding:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1,
    stratify=y
)

st.write("Jumlah data training:", X_train.shape[0])
st.write("Jumlah data testing:", X_test.shape[0])

st.header("3. Training Model")

dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=0
)

dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    random_state=0,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

st.success("Training kedua model berhasil!")

y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

st.header("4. Evaluasi dan Perbandingan Model")

def evaluate_model(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "F1-Score": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
    }

dt_results = evaluate_model(y_test, y_pred_dt)
rf_results = evaluate_model(y_test, y_pred_rf)

comparison = pd.DataFrame(
    [dt_results, rf_results],
    index=["Decision Tree", "Random Forest"]
)

col1, col2 = st.columns(2)

col1.metric(
    "Decision Tree Accuracy",
    f"{dt_results['Accuracy'] * 100:.2f}%"
)

col2.metric(
    "Random Forest Accuracy",
    f"{rf_results['Accuracy'] * 100:.2f}%"
)

st.subheader("Tabel Perbandingan Performa")

st.dataframe(
    (comparison * 100).round(2).rename_axis("Model"),
    width="stretch"
)

st.subheader("Grafik Perbandingan Algoritma")

fig, ax = plt.subplots(figsize=(10, 5))

comparison.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Perbandingan Decision Tree dan Random Forest")
ax.set_xlabel("Algoritma")
ax.set_ylabel("Nilai Evaluasi")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right")
ax.tick_params(axis="x", rotation=0)

st.pyplot(fig)
plt.close(fig)

if dt_results["Accuracy"] > rf_results["Accuracy"]:
    best_model = "Decision Tree"
elif rf_results["Accuracy"] > dt_results["Accuracy"]:
    best_model = "Random Forest"
else:
    best_model = "Keduanya memiliki akurasi yang sama"

st.info(f"Model dengan akurasi tertinggi: **{best_model}**")

st.header("5. Confusion Matrix")

col1, col2 = st.columns(2)

cm_dt = confusion_matrix(
    y_test, y_pred_dt,
    labels=np.arange(len(class_names))
)

cm_rf = confusion_matrix(
    y_test, y_pred_rf,
    labels=np.arange(len(class_names))
)

with col1:
    st.subheader("Decision Tree")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_dt,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Random Forest")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_rf,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

st.header("6. Classification Report")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Decision Tree")
    report_dt = classification_report(
        y_test,
        y_pred_dt,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    st.dataframe(
        pd.DataFrame(report_dt).transpose().round(4),
        width="stretch"
    )

with col2:
    st.subheader("Random Forest")
    report_rf = classification_report(
        y_test,
        y_pred_rf,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    st.dataframe(
        pd.DataFrame(report_rf).transpose().round(4),
        width="stretch"
    )

st.header("7. Feature Importance Random Forest")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).head(15)

fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature",
    ax=ax
)

ax.set_title("Top 15 Feature Importance - Random Forest")
st.pyplot(fig)
plt.close(fig)

st.header("8. Visualisasi Decision Tree")

fig, ax = plt.subplots(figsize=(20, 10))

plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=7,
    ax=ax
)

ax.set_title("Decision Tree - Gini (max_depth=3)")
st.pyplot(fig)
plt.close(fig)

st.header("9. Kesimpulan")

st.write(
    f"""
    Berdasarkan hasil pengujian menggunakan dataset Mushroom Classification:

    - Decision Tree memperoleh accuracy sebesar
      {dt_results['Accuracy'] * 100:.2f}%.
    - Random Forest memperoleh accuracy sebesar
      {rf_results['Accuracy'] * 100:.2f}%.
    - Model dengan accuracy tertinggi adalah **{best_model}**.

    Perbandingan ini menunjukkan performa kedua algoritma
    berdasarkan dataset dan pembagian data pengujian yang digunakan.
    """
)

st.caption("Mushroom Classification | Decision Tree vs Random Forest")
