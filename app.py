import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load Dataset & Model

@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease_uci.csv") 
    return df

@st.cache_resource
def load_model():
    model = joblib.load("best_model.joblib")  
    return model

df = load_data()
model = load_model()


# App Title & Description

st.set_page_config(page_title="ML Prediction App", layout="wide")
st.title("Heart Disease Prediction App")
st.markdown("""
This interactive Streamlit app lets you:
- Explore the dataset
- Visualize trends & patterns
- Make predictions using a trained ML model
- View model performance and comparisons
""")

# Sidebar Navigation

menu = st.sidebar.radio(
    "Navigate to:",
    ["Data Exploration", "Visualization", "Model Prediction", "Model Performance"]
)


# Data Exploration

if menu == "Data Exploration":
    st.header("Dataset Overview")
    st.write(f"**Shape:** {df.shape}")
    st.write("**Column Data Types:**")
    st.dataframe(df.dtypes.rename("Data Type"))

    st.subheader("Sample Data")
    st.dataframe(df.sample(10))

    st.subheader("Interactive Filtering")
    if "age" in df.columns:
        min_age, max_age = int(df["age"].min()), int(df["age"].max())
        age_range = st.slider("Select Age Range:", min_age, max_age, (min_age, max_age))
        filtered_df = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]
        st.write(f"Filtered rows: {filtered_df.shape[0]}")
        st.dataframe(filtered_df)

# Visualization

elif menu == "Visualization":
    st.header("Data Visualizations")

    # Histogram
    if "age" in df.columns:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["age"], bins=20, ax=ax, color="skyblue")
        st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)  # select only numeric columns
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    # Interactive Scatter Plot
    st.subheader("Interactive Scatter Plot")
    x_axis = st.selectbox("Select X-axis:", df.columns, index=0)
    y_axis = st.selectbox("Select Y-axis:", df.columns, index=1)
    color_by = st.selectbox("Color by:", df.columns, index=2)
    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by)
    st.plotly_chart(fig)

# Model Prediction

elif menu == "Model Prediction":
    st.header("Make a Prediction")

    # Select all features except target
    features = [col for col in df.columns if col != "target"]  # <-- change 'target'
    user_inputs = {}

    st.markdown("### Enter feature values:")
    for col in features:
        if np.issubdtype(df[col].dtype, np.number):
            user_inputs[col] = st.number_input(
                f"{col}", 
                float(df[col].min()), 
                float(df[col].max()), 
                float(df[col].mean())
            )
        else:
            user_inputs[col] = st.selectbox(f"{col}", df[col].unique())

    input_df = pd.DataFrame([user_inputs])

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            try:
                pred = model.predict(input_df)[0]
                st.success(f"Prediction: {pred}")

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_df)
                    confidence = np.max(proba) * 100
                    st.info(f"Prediction Confidence: {confidence:.2f}%")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# Model Performance

# Model Performance
elif menu == "Model Performance":
    st.header("Model Performance")

    try:
        # Use raw strings for Windows paths and updated filenames
        X_test = pd.read_csv(r"X_test.csv")
        y_test = pd.read_csv(r"y_test.csv").values.ravel()

        # Optional: check if model is fitted
        from sklearn.utils.validation import check_is_fitted
        try:
            check_is_fitted(model)
        except:
            st.error("The loaded model is not fitted. Please retrain and save a fitted model.")
            st.stop()

        with st.spinner("Evaluating model..."):
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f"**Accuracy:** {acc:.2%}")

            st.subheader("Classification Report")
            report_df = pd.DataFrame(
                classification_report(y_test, y_pred, output_dict=True)
            ).transpose()
            st.dataframe(report_df)

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            st.subheader("Model Comparison")
            st.write("You can add multiple model scores here for comparison.")

    except FileNotFoundError:
        st.error("Test data files not found. Please run the training notebook to generate X_test.csv and y_test.csv.")

