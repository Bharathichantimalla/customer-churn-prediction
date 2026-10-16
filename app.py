
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 1. Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System")
st.write(
    "Analyze customer data and predict whether a "
    "telecom customer is likely to leave the service."
)

# -------------------------------
# 2. Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("customer_churn.csv")

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    return df


# -------------------------------
# 3. Train Machine Learning Model
# -------------------------------
@st.cache_resource
def train_model(df):
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    numeric_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        target_names=["Stayed", "Churned"],
        output_dict=True,
        zero_division=0
    )

    return model, accuracy, report


# -------------------------------
# 4. Main Application
# -------------------------------
try:
    df = load_data()

    # Dataset Overview
    st.success("Dataset loaded successfully!")

    st.header("1. Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric(
        "Overall Churn Rate",
        f"{(df['Churn'] == 'Yes').mean() * 100:.1f}%"
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Churn Visualization
    st.header("2. Customer Churn Analysis")

    churn_counts = df["Churn"].value_counts()

    fig, ax = plt.subplots()
    churn_counts.plot(kind="bar", ax=ax)

    ax.set_title("Customers Who Stayed vs Left")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")
    ax.tick_params(axis="x", rotation=0)

    st.pyplot(fig)
    plt.close(fig)

    # Model Training
    st.header("3. Machine Learning Model")

    with st.spinner("Training the model..."):
        model, accuracy, report = train_model(df)

    st.success("Model trained successfully!")

    st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")

    st.subheader("Model Performance Report")
    st.dataframe(
        pd.DataFrame(report).transpose().round(3),
        use_container_width=True
    )

    st.caption(
        "Accuracy is measured on a held-out test dataset. "
        "Actual performance may differ on new customers."
    )

    # -------------------------------
    # 5. Customer Churn Prediction Form
    # -------------------------------
    st.header("4. Predict Customer Churn")

    st.write("Enter customer details below.")

    with st.form("churn_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Gender", ["Female", "Male"]
            )

            senior = st.selectbox(
                "Senior Citizen", [0, 1]
            )

            partner = st.selectbox(
                "Has Partner?", ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Has Dependents?", ["Yes", "No"]
            )

            tenure = st.slider(
                "Tenure (months)", 0, 72, 12
            )

            phone = st.selectbox(
                "Phone Service", ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

        with col2:
            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless = st.selectbox(
                "Paperless Billing", ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=200.0,
                value=70.0
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=10000.0,
                value=840.0
            )

        submitted = st.form_submit_button(
            "Predict Churn"
        )

    # -------------------------------
    # 6. Show Prediction
    # -------------------------------
    if submitted:

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        prediction = model.predict(customer)[0]

        probabilities = model.predict_proba(customer)[0]
        churn_probability = probabilities[1]

        if prediction == 1:
            st.error(
                "⚠️ Prediction: Customer may leave the service."
            )
        else:
            st.success(
                "✅ Prediction: Customer is likely to stay."
            )

        st.metric(
            "Estimated Churn Probability",
            f"{churn_probability * 100:.2f}%"
        )

        st.caption(
            "This is a machine learning estimate, not a guarantee."
        )

except FileNotFoundError:
    st.error(
        "Dataset not found. Place customer_churn.csv "
        "in the same folder as app.py."
    )

except Exception as e:
    st.error(f"Something went wrong: {e}")
