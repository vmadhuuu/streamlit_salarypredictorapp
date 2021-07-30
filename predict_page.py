import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    html_temp = """
    <div style="background-color:#5C1421 ;padding:10px;border-radius:20px;margin:10px">
    <h1 style="color:white;text-align:center;">Software Developer Salary Predictor </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.write("""### Choose from the options: """)

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    experience = st.slider("Years of Experience", 0, 30, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, experience ]])
        X[:, 0] = le_country.fit_transform(X[:,0])
        X[:, 1] = le_education.fit_transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
