import json

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

f = open("data.json", "r")
data = json.load(f)

plt.rcParams['font.family'] = 'poppins'


# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    if np.max(predictions) < 0.3:
        return -1
    return np.argmax(predictions)  # return index of max element


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if app_mode == "Home":
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = "home_img.jpg"
    st.image(image_path)

# About Project
elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, width=4, use_column_width=True)
    # Predict button
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        # Reading Labels
        with open("labels.txt") as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])
        if result_index == -1:
            st.error("Not a fruit!")
        else:
            fruit = data[str(result_index)]
            st.success("Model is Predicting it's a {}".format(fruit['name']))
            st.header("Nutrition Facts")
            st.divider()
            st.text(f"Serving Size: {fruit['serving_size']}")
            st.divider()
            st.subheader(f"Calories: {fruit['calories']}")
            st.divider()
            nutrition = fruit['nutrition']
            made_of = fruit['made_of']

            df = pd.DataFrame(nutrition, columns=['Name', '', '% Daily Value*'])

            st.table(df)
            st.write("*The % Daily Value (DV) tells you how much a nutrient in a serving of food contributes to a "
                     "daily diet. 2,000 calories a day is used for general nutrition advice.")

            fig, ax = plt.subplots()
            percent = np.array(made_of['percents'])
            ax.pie(percent, colors=made_of['colors'], autopct='%1.1f%%', labeldistance=0.5,
                   counterclock=False, wedgeprops={"linewidth": 2, "edgecolor": "white"},
                   textprops={'color': "#333333"})
            legend = [f'{percent:05.2f}%    {label}' for percent, label in zip(made_of['percents'], made_of['labels'])]

            plt.legend(labels=legend, loc="best")
            ax.axis('equal')
            st.subheader("WHAT IS THIS FOOD MADE OF?")
            st.pyplot(fig)
