# Handwritten Digit Generator with Streamlit and PyTorch

This project is a web application that generates images of handwritten digits. It leverages a Conditional Generative Adversarial Network (cGAN) built with PyTorch and trained on the famous MNIST dataset. 

Users can select a specific digit (from 0 to 9), and the model will generate five unique images corresponding to that selection. The entire application is deployed as an interactive web app using Streamlit.

## Live Demo

**(Don't forget to replace the link below with your actual Streamlit App URL!)**

You can access and interact with the live application here: 
**[https://answer3-k5f4efgbuyjt7yjzcwcwuc.streamlit.app/](https://answer3-k5f4efgbuyjt7yjzcwcwuc.streamlit.app/)**

## Features

-   User-friendly web interface built with Streamlit.
-   Allows users to select a specific digit (0-9) for generation.
-   Generates 5 unique images for the selected digit on demand.
-   Utilizes a Conditional GAN (cGAN) to control the generated output based on user input.
-   The model was trained from scratch on the MNIST dataset using Google Colab.

## Technologies Used

-   **Core Language:** Python
-   **Model/Training:** PyTorch
-   **Dataset:** MNIST
-   **Web Framework:** Streamlit
-   **Deployment:** Streamlit Community Cloud

## How to Run Locally

To run this application on your own machine, please follow these steps:

1.  **Clone the repository:**
    ```bash
    # Replace the URL with your own repository's URL
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install the required dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in a new tab in your web browser.

## Project Structure
