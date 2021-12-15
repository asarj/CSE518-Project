import numpy as np
import tensorflow as tf
from keract import get_activations, display_activations
import streamlit as st
from skimage.color import rgb2gray
import skimage.transform as ski
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from pprint import pprint
from os import listdir
from os.path import isfile, join
import shutil

# Config vars
st.set_page_config(layout="wide")
ACTIVATIONS_PATH = "./activations"

# Get the model
# labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
labels = map(lambda x: x.strip('\r').split(' '), open('./emnist-byclass-mapping.txt').read().strip().split('\n'))
labels = dict(labels)
# pprint(labels)
model = tf.keras.models.load_model("./experimentation/final_model")

# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 23)
stroke_color = "#FFFFFF"
bg_color = "#000000"
canvas_css = """
.stMarkdown {
    display: grid;
    place-items: center;
}
.stButton {
    display: grid;
    place-items: center;
}

.canvas-container {
    display: grid;
    place-items: center;
}
"""

def label_to_char(label):
  ascii_val = labels[str(label)]
  return chr(int(ascii_val))


def preprocess_input_img(img: np.ndarray):
    img = img.astype(np.float32)
    img = rgb2gray(ski.rescale(img, 14/225, multichannel=True))
    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, 0)
    img = img / 255.0
    return img


def get_pred_data(img: np.ndarray):
    return model.predict(preprocess_input_img(img))


def generate_model_viz(image: np.ndarray):
    layer_outputs = [layer.output for layer in model.layers] # Extracts the outputs of the top 12 layers
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
    activations = activation_model.predict(preprocess_input_img(image)) # Returns a list of five Numpy arrays: one array per layer activation

    layer_names = []
    for layer in model.layers[:4]:
        layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
        
    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
        n_features = layer_activation.shape[-1] # Number of features in the feature map
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, # Displays the grid
                            row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        fig = plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        # fig.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        st.markdown(f"### {layer_name}")
        # fig.title ( layer_name )
        # fig.grid  ( False )
        # plt.imshow( image_belt, aspect='auto')
        st.pyplot(fig)


if __name__ == "__main__":
    # print(model.summary())

    st.title("Visualizing Machine Learning Model Outputs")
    c1, c2 = st.columns((50,50))
    # Create a canvas component
    # st.markdown(f'<style>{canvas_css}</style>', unsafe_allow_html=True)  # to center title for reference
    predict = False
    canvas_result = None
    img = None
    with c1:
        preds = None
        top_pred = None
        predict = False
        # canvas_result.json_data.objects = []
        canvas_result = None
        st.markdown("## Draw an Alphanumeric Character")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=None,
            update_streamlit=True,
            height=450,
            width=450,
            drawing_mode="freedraw",
            key="canvas",
        )
        predict = st.button("Predict")
    with c2:
        if predict:
            fig = plt.figure(figsize=(28,28), frameon=False)
            img = canvas_result.image_data
            # print("RAW IMG SHAPE: ", img.shape)
            # img = rgb2gray(ski.rescale(img, 14/225, multichannel=True))
            # img = np.expand_dims(img, axis=2)
            # img = np.expand_dims(img, 0)
            # print("NEW IMG SHAPE: ", img.shape)
            # plt.imshow(img[0], cmap="gray")
            # fig.savefig("fig.png")
            preds = get_pred_data(img=img)
            print(f"Prediction value: {str(np.argmax(preds, 1)[0])}")
            print(f"Old code value: {label_to_char(np.argmax(preds, 1)[0])}")
            print(f"Mapped Prediction value: {label_to_char(np.argmax(preds, 1)[0])}")
            top_pred = label_to_char(np.argmax(preds, 1)[0])
            st.markdown("##              ")
            st.markdown("##              ")
            st.markdown("##              ")
            st.markdown(f"## Prediction Result: {top_pred}")
            # st.markdown(f"# ")
            probabilities, top_k_preds = tf.nn.top_k(preds, k=5)
            top_k_preds = list(top_k_preds.numpy()[0])
            probabilities = list(probabilities.numpy()[0])
            st.markdown(f"### Top 5 Probabilities")
            for i in range(len(top_k_preds)):
                ascii_char = label_to_char(top_k_preds[i])
                probability = probabilities[i]
                st.markdown(f"{ascii_char}: {probability * 100:.2f}%")
            start_new = st.button("Start New")
            if start_new:
                preds = None
                top_pred = None
                predict = False
                canvas_result.json_data.objects = []
                canvas_result = None

    if predict:
        st.markdown("## What the Model Saw")
        # generate_model_viz(img)

        # st.markdown("## More on Activation Layers")
        activations = get_activations(model, preprocess_input_img(img), auto_compile=True)

        with st.spinner("Getting Model Activation Layers Ready..."):
            display_activations(activations, cmap=None, save=True, directory='./activations', data_format='channels_last', fig_size=(24, 12), reshape_1d_layers=False)

        onlyfiles = [f for f in listdir(ACTIVATIONS_PATH) if isfile(join(ACTIVATIONS_PATH, f))]
        onlyfiles = sorted(onlyfiles)
        for f in onlyfiles:
            split_on_dot = f.split(".")
            split_on_u = split_on_dot[0].split("_")
            st.markdown(f"### {'_'.join(split_on_u[1:])}")
            st.image(ACTIVATIONS_PATH + "/" + f)
        shutil.rmtree(ACTIVATIONS_PATH)
        preds = None
        top_pred = None
        predict = False
        # canvas_result.json_data.objects = []
        canvas_result = None