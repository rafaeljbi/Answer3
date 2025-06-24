import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# --- Definição da Arquitetura do Modelo ---
# Esta classe DEVE ser exatamente a mesma que você usou no treinamento.
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

# --- Funções da Aplicação ---
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
st.set_page_config(page_title="Gerador de Dígitos", layout="wide")
st.title("Gerador de Dígitos Manuscritos com IA")
st.write("Aplicação para gerar imagens de dígitos manuscritos usando uma GAN treinada no dataset MNIST.")

generator_model = load_model()

st.selectbox('Selecione um dígito (este seletor é visual, a GAN gera aleatoriamente):', list(range(10)))

if st.button('Gerar 5 Novas Imagens'):
    with st.spinner('Gerando imagens...'):
        images = generate_images(generator_model, 5)
        numpy_images = [img.squeeze().detach().numpy() for img in images]
        st.subheader("Imagens Geradas:")
        cols = st.columns(5)
        for i, image_np in enumerate(numpy_images):
            cols[i].image(image_np, caption=f'Imagem {i+1}', width=150)
else:
    st.info("Clique no botão acima para gerar imagens.")