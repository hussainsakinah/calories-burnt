import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import zipfile
import os

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold   
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score                    

warnings.filterwarnings('ignore')

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Calories Burnt Predictor",
    page_icon="🔥",
    layout="wide",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF4500;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #333;
    }
    .prediction-box {
        background: linear-gradient(135deg, #FF4500, #ff7043);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🔥 Calories Burnt Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ridge Regression · Polynomial Features · GridSearchCV</div>', unsafe_allow_html=True)

# ─── Data Loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_data(calories_file, exercise_file):
    calories_df = pd.read_csv(calories_file)
    exercise_df = pd.read_csv(exercise_file)
    df = pd.concat([exercise_df, calories_df["Calories"]], axis=1)
    return df

@st.cache_resource
def train_model(df):
    data = df.copy()
    label_encoder = preprocessing.LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])

    X = data.drop(columns=["User_ID", "Calories"])
    Y = data["Calories"]

    # ── Train/test split (fit ONLY on training data) ──────────────────────────
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    
    ridge_pipeline = Pipeline([
        ('poly',   PolynomialFeatures(include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge',  Ridge())
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)


    param_grid_ridge = {
        'poly__degree': [1, 2],            
        'ridge__alpha': np.geomspace(1e-3, 1e3, 20), 
    }

    grid_ridge = GridSearchCV(
        ridge_pipeline, param_grid_ridge,
        cv=kf, scoring='r2', n_jobs=-1
    )
    grid_ridge.fit(X_train, Y_train)

    best_ridge = grid_ridge.best_estimator_

    # Evaluate on held-out test set AND training set
    y_pred       = best_ridge.predict(X_test)
    y_pred_train = best_ridge.predict(X_train)

    mse      = mean_squared_error(Y_test, y_pred)
    r2_test  = r2_score(Y_test,  y_pred)
    r2_train = r2_score(Y_train, y_pred_train)
    cv_r2    = grid_ridge.best_score_

    return best_ridge, grid_ridge.best_params_, mse, r2_test, r2_train, cv_r2, X_test, Y_test, y_pred, label_encoder

# ─── Sidebar: Upload CSVs ────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/fire-element--v1.png", width=80)
st.sidebar.title("📂 Upload Data")
st.sidebar.markdown("Upload your **calories.csv** and **exercise.csv** files.")

calories_file  = st.sidebar.file_uploader("Upload calories.csv",  type="csv", key="cal")
exercise_file  = st.sidebar.file_uploader("Upload exercise.csv",  type="csv", key="ex")

st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.info(
    "This app uses a Ridge Regression pipeline with Polynomial Features "
    "and GridSearchCV to predict calories burnt during exercise."
)

# ─── Main Logic ──────────────────────────────────────────────────────────────
if calories_file and exercise_file:

    df = load_data(calories_file, exercise_file)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview",
        "📈 Visualizations",
        "🤖 Model Performance",
        "🔥 Predict Calories"
    ])

    # ── Tab 1: Data Overview ─────────────────────────────────────────────────
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", f"{df.shape[0]:,}")
        col2.metric("Features",      f"{df.shape[1] - 2}")  # excluding User_ID & Calories
        col3.metric("Missing Values", f"{df.isnull().sum().sum()}")

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("Data Types & Null Counts")
        info_df = pd.DataFrame({
            "Data Type": df.dtypes,
            "Null Count": df.isnull().sum(),
            "Null %": (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(info_df, use_container_width=True)

    # ── Tab 2: Visualizations ────────────────────────────────────────────────
    with tab2:
        st.subheader("Gender Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(x=df["Gender"], ax=ax, palette="Oranges")
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        st.pyplot(fig)
        plt.close()

        st.subheader("Feature Distributions")
        numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
        cols = st.columns(2)
        for i, col_name in enumerate(numeric_cols):
            with cols[i % 2]:
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.histplot(df[col_name], kde=True, ax=ax, color="#FF4500")
                ax.set_title(col_name, color="white")
                ax.set_facecolor("#0e1117")
                fig.patch.set_facecolor("#0e1117")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                st.pyplot(fig)
                plt.close()

        st.subheader("Correlation Heatmap")
        data_enc = df.copy()
        le = preprocessing.LabelEncoder()
        data_enc['Gender'] = le.fit_transform(data_enc['Gender'])
        data_enc = data_enc.drop(columns=["User_ID"])
        correlation = data_enc.corr()

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            correlation, cbar=True, square=True, fmt='.1f',
            annot=True, annot_kws={'size': 9}, cmap='Oranges', ax=ax
        )
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        st.pyplot(fig)
        plt.close()

        st.subheader("Skewness")
        skewness = data_enc.skew().reset_index()
        skewness.columns = ["Feature", "Skewness"]
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=skewness, x="Feature", y="Skewness", palette="Oranges_r", ax=ax)
        ax.axhline(0, color="white", linewidth=0.8, linestyle="--")
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        plt.xticks(rotation=30)
        st.pyplot(fig)
        plt.close()

    # ── Tab 3: Model Performance ─────────────────────────────────────────────
    with tab3:
        with st.spinner("Training Ridge Regression model (this may take ~30 seconds)…"):
            best_ridge, best_params, mse, r2_test, r2_train, cv_r2, X_test, Y_test, y_pred, label_encoder = train_model(df)

        st.subheader("Best Hyperparameters")
        st.dataframe(pd.DataFrame([best_params]), use_container_width=True)


        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏋️ Train R²",        f"{r2_train:.6f}")
        col2.metric("🧪 Test R²",         f"{r2_test:.6f}")
        col3.metric("📊 5-Fold CV R²",    f"{cv_r2:.6f}")
    
        rmse = float(np.sqrt(mse))
        mae  = float(np.mean(np.abs(Y_test.values - y_pred)))

        st.subheader("Error Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("📏 RMSE (kcal)", f"{rmse:.4f}")
        col2.metric("📊 MAE  (kcal)", f"{mae:.4f}")
        col3.metric("📈 R²",          f"{r2_test:.4f}")

        st.subheader("Actual vs Predicted Calories")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(Y_test, y_pred, alpha=0.4, color="#FF4500", edgecolors="white", linewidths=0.3)
        lims = [min(Y_test.min(), y_pred.min()), max(Y_test.max(), y_pred.max())]
        ax.plot(lims, lims, 'w--', linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual Calories",    color="white")
        ax.set_ylabel("Predicted Calories", color="white")
        ax.set_title("Actual vs Predicted", color="white")
        ax.legend(facecolor="#222", labelcolor="white")
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        st.pyplot(fig)
        plt.close()

        st.subheader("Residuals")
        residuals = Y_test.values - y_pred
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(residuals, kde=True, ax=ax, color="#FF4500")
        ax.axvline(0, color="white", linestyle="--")
        ax.set_xlabel("Residual", color="white")
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        st.pyplot(fig)
        plt.close()

    # ── Tab 4: Predict ───────────────────────────────────────────────────────
    with tab4:
        st.subheader("🔥 Predict Calories Burnt")
        st.markdown("Enter your details below and hit **Predict**.")

       
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                gender     = st.selectbox("Gender",     ["male", "female"])
                age        = st.number_input("Age (years)",       min_value=1,   max_value=120, value=25)
                height     = st.number_input("Height (cm)",       min_value=50.0, max_value=250.0, value=170.0, step=0.5)
                body_temp  = st.number_input("Body Temp (°C)",    min_value=35.0, max_value=45.0, value=37.0, step=0.1)
            with col2:
                weight     = st.number_input("Weight (kg)",       min_value=10.0, max_value=300.0, value=70.0, step=0.5)
                duration   = st.number_input("Duration (minutes)",min_value=0.0, max_value=300.0, value=30.0, step=1.0)
                heart_rate = st.number_input("Heart Rate (bpm)",  min_value=30.0, max_value=220.0, value=100.0, step=1.0)

            submitted = st.form_submit_button("🔥 Predict Calories", use_container_width=True)

        if submitted:
           
            with st.spinner("Predicting…"):
                best_ridge, best_params, mse, r2_test, r2_train, cv_r2, X_test, Y_test, y_pred, label_encoder = train_model(df)

                gender_encoded = label_encoder.transform([gender])[0]
                input_data = np.array([[gender_encoded, age, height, weight, duration, heart_rate, body_temp]])

                prediction = best_ridge.predict(input_data)[0]

            st.markdown(
                f'<div class="prediction-box">🔥 Estimated Calories Burnt: <strong>{prediction:.2f} kcal</strong></div>',
                unsafe_allow_html=True
            )
            st.balloons()

else:
    st.info("👈 Please upload both **calories.csv** and **exercise.csv** from the sidebar to get started.")
    st.markdown("""
    ### How to use this app
    1. **Upload** your `calories.csv` and `exercise.csv` files using the sidebar.
    2. Explore the **Data Overview** and **Visualizations** tabs.
    3. Check model performance in the **Model Performance** tab.
    4. Enter your stats in the **Predict Calories** tab to get a prediction.

    > The dataset should have columns:  
    > **exercise.csv** → `User_ID, Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp`  
    > **calories.csv** → `User_ID, Calories`
    """)
