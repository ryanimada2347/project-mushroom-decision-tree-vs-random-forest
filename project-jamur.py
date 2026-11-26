
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)

st.set_page_config(
    page_title="Mushroom Classification",
    page_icon="🍄",
    layout="wide"
)

st.title("🍄 Mushroom Classification")
st.subheader("Decision Tree Classifier")
st.write(
    "Klasifikasi jamur berdasarkan karakteristik "
    "untuk menentukan apakah jamur edible atau poisonous."
)

st.divider()

st.header("1. Dataset")

uploaded_file = st.file_uploader(
    "Upload dataset mushrooms.csv (opsional)",
    type=["csv"]
)

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
else:
    dataset_path = Path(__file__).parent / "mushrooms.csv"

    if not dataset_path.exists():
        st.warning(
            "File mushrooms.csv tidak ditemukan. "
            "Silakan upload dataset terlebih dahulu."
        )
        st.stop()

    dataset = pd.read_csv(dataset_path)

st.success("Dataset berhasil dimuat!")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Jumlah Data", dataset.shape[0])

with col2:
    st.metric("Jumlah Kolom", dataset.shape[1])

with col3:
    st.metric(
        "Missing Values",
        dataset.isnull().sum().sum()
    )

st.subheader("Preview Dataset")
st.dataframe(dataset.head(10), use_container_width=True)

st.header("2. Exploratory Data Analysis")

if "class" not in dataset.columns:
    st.error("Kolom 'class' tidak ditemukan dalam dataset.")
    st.stop()

st.write("Distribusi kelas jamur:")

class_counts = dataset["class"].value_counts()

fig, ax = plt.subplots(figsize=(8, 4))

sns.histplot(
    data=dataset,
    x="class",
    discrete=True,
    ax=ax
)

ax.set_title("Distribusi Kelas Jamur")
ax.set_xlabel("Class")
ax.set_ylabel("Jumlah Data")

st.pyplot(fig)
plt.close(fig)

col1, col2 = st.columns(2)

with col1:
    st.write("**Nilai unik pada kolom class**")
    st.write(dataset["class"].unique())

with col2:
    st.write("**Jumlah masing-masing kelas**")
    st.dataframe(class_counts)

st.header("3. Data Preprocessing")

X = dataset.drop(["class"], axis=1)
y = dataset["class"]

X = pd.get_dummies(X)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

st.write("**Fitur setelah proses One-Hot Encoding**")
st.dataframe(X.head(), use_container_width=True)

st.write("**Label hasil Label Encoding**")
st.write(encoder.classes_)

st.write("Jumlah fitur setelah encoding:", X.shape[1])

st.header("4. Data Splitting")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1,
    stratify=y
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Training Data", X_train.shape[0])

with col2:
    st.metric("Testing Data", X_test.shape[0])

st.write("Training Features:", X_train.shape)
st.write("Testing Features:", X_test.shape)

st.header("5. Modeling Decision Tree")

max_depth = st.slider(
    "Pilih Maximum Depth",
    min_value=1,
    max_value=10,
    value=3
)

clf_gini = DecisionTreeClassifier(
    criterion="gini",
    max_depth=max_depth,
    random_state=0
)

clf_gini.fit(X_train, y_train)

clf_en = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=max_depth,
    random_state=0
)

clf_en.fit(X_train, y_train)

y_pred_gini = clf_gini.predict(X_test)
y_pred_train_gini = clf_gini.predict(X_train)

y_pred_en = clf_en.predict(X_test)
y_pred_train_en = clf_en.predict(X_train)

st.success("Kedua model Decision Tree berhasil dilatih!")

st.header("6. Model Evaluation")

accuracy_gini = accuracy_score(y_test, y_pred_gini)
accuracy_en = accuracy_score(y_test, y_pred_en)

train_accuracy_gini = accuracy_score(
    y_train, y_pred_train_gini
)

train_accuracy_en = accuracy_score(
    y_train, y_pred_train_en
)

f1_gini = f1_score(y_test, y_pred_gini)
f1_en = f1_score(y_test, y_pred_en)

comparison = pd.DataFrame({
    "Model": ["Gini", "Entropy"],
    "Training Accuracy": [
        train_accuracy_gini,
        train_accuracy_en
    ],
    "Testing Accuracy": [
        accuracy_gini,
        accuracy_en
    ],
    "F1 Score": [
        f1_gini,
        f1_en
    ]
})

st.subheader("Perbandingan Model")

st.dataframe(
    comparison.style.format({
        "Training Accuracy": "{:.4f}",
        "Testing Accuracy": "{:.4f}",
        "F1 Score": "{:.4f}"
    }),
    use_container_width=True
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Accuracy Gini",
        f"{accuracy_gini:.4f}"
    )

with col2:
    st.metric(
        "Accuracy Entropy",
        f"{accuracy_en:.4f}"
    )

st.subheader("Detail Evaluasi Model")

selected_model = st.selectbox(
    "Pilih Model Decision Tree",
    ["Gini", "Entropy"]
)

if selected_model == "Gini":
    selected_clf = clf_gini
    selected_pred = y_pred_gini
else:
    selected_clf = clf_en
    selected_pred = y_pred_en

st.write("### Confusion Matrix")

cm = confusion_matrix(y_test, selected_pred)

fig, ax = plt.subplots(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    linewidths=0.5,
    linecolor="red",
    fmt=".0f",
    cmap="Blues",
    xticklabels=["Edible", "Poisonous"],
    yticklabels=["Edible", "Poisonous"],
    ax=ax
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_title(f"Confusion Matrix - {selected_model}")

st.pyplot(fig)
plt.close(fig)

st.write("### Classification Report")

report = classification_report(
    y_test,
    selected_pred,
    target_names=["Edible", "Poisonous"],
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.style.format("{:.4f}"),
    use_container_width=True
)

selected_f1 = f1_score(y_test, selected_pred)

st.metric(
    "F1 Score",
    f"{selected_f1:.4f}"
)

st.header("7. Decision Tree Visualization")

fig, ax = plt.subplots(figsize=(22, 10))

plot_tree(
    selected_clf,
    feature_names=X.columns,
    class_names=["Edible", "Poisonous"],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax
)

ax.set_title(
    f"Decision Tree - {selected_model} "
    f"(max_depth={max_depth})"
)

st.pyplot(fig)
plt.close(fig)

st.divider()

st.caption(
    "Mushroom Classification | Decision Tree "
    "using Gini Index and Entropy"
)
