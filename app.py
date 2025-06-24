import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# --- Archtecture ---
# This class should be the same used to train the model
LATENT_SIZE = 64
HIDDEN_SIZE = 256
IMAGE_SIZE = 784  # 28x28

class Generator(nn.Module):
    def __init__(self, latent_size, hidden_size, image_size):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# --- Functions of application ---
@st.cache_resource
def load_model():
    """Carrega o gerador treinado a partir do arquivo .ckpt"""
    model = Generator(LATENT_SIZE, HIDDEN_SIZE, IMAGE_SIZE)
    model.load_state_dict(torch.load('generator.ckpt', map_location=torch.device('cpu')))
    model.eval()
    return model

def generate_images(model, num_images=5):
    """Gera imagens usando o modelo."""
    z = torch.randn(num_images, LATENT_SIZE)
    generated_images = model(z)
    generated_images = generated_images.view(num_images, 1, 28, 28)
    generated_images = (generated_images + 1) / 2
    return generated_images

# --- Interface Gráfica do Streamlit ---
st.set_page_config(page_title="Handwritten Digit generator", layout="wide")
st.title("Hand-written Digit generator with AI")
st.write("App that generates handwritten digit images using a GAN application trained on MNIST.")

generator_model = load_model()

st.selectbox('Select a digit (This selector is visual, the GAN generates randomly)):', list(range(10)))

if st.button('Generate  new images'):
    with st.spinner('Generating images...'):
        images = generate_images(generator_model, 5)
        numpy_images = [img.squeeze().detach().numpy() for img in images]
        st.subheader("generated Images:")
        cols = st.columns(5)
        for i, image_np in enumerate(numpy_images):
            cols[i].image(image_np, caption=f'Imagem {i+1}', width=150)
else:
    st.info("Click on the button above to generate images.")