import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# --- Conditional Architecture ---
# This class MUST be identical to the one used in the cGAN training script.
LATENT_SIZE = 100
HIDDEN_SIZE = 256
IMAGE_SIZE = 784
NUM_CLASSES = 10
EMBEDDING_SIZE = 10

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(NUM_CLASSES, EMBEDDING_SIZE)
        self.main = nn.Sequential(
            nn.Linear(LATENT_SIZE + EMBEDDING_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, IMAGE_SIZE),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Concatenate noise vector (z) with the label embedding (c)
        c = self.label_embedding(labels)
        x = torch.cat([z, c], 1)
        return self.main(x)

# --- Application Functions ---
@st.cache_resource
def load_model():
    """Loads the trained cGAN generator."""
    model = Generator()
    # ATTENTION: Load the new cGAN checkpoint file
    model.load_state_dict(torch.load('cgan_generator.ckpt', map_location=torch.device('cpu')))
    model.eval() # Set the model to evaluation mode
    return model

def generate_images(model, digit_label, num_images=5):
    """Generates images for a specific digit."""
    # Create random noise
    z = torch.randn(num_images, LATENT_SIZE)
    # Create labels for the chosen digit
    labels = torch.LongTensor([digit_label] * num_images)
    # Generate images using the model with both noise and labels
    generated_images = model(z, labels)
    generated_images = generated_images.view(num_images, 1, 28, 28)
    generated_images = (generated_images + 1) / 2 # Denormalize
    return generated_images

# --- Streamlit User Interface ---
st.set_page_config(page_title="Conditional Digit Generator", layout="wide")
st.title("On-Demand Digit Generator with cGAN")
st.write("Select a digit and the AI will generate corresponding images.")

generator_model = load_model()

# Capture the user's choice in a variable
digit_choice = st.selectbox('Select the digit you want to generate:', list(range(10)))

if st.button(f'Generate 5 images of digit {digit_choice}'):
    with st.spinner(f'Generating images of digit {digit_choice}...'):
        # Pass the user's choice to the generation function
        images = generate_images(generator_model, digit_choice, 5)
        numpy_images = [img.squeeze().detach().numpy() for img in images]
        st.subheader(f"Images Generated for Digit {digit_choice}:")
        cols = st.columns(5)
        for i, image_np in enumerate(numpy_images):
            cols[i].image(image_np, caption=f'Generated {i+1}', width=150)
else:
    st.info("Click the button above to generate images.")