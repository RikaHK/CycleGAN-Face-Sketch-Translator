# Assignment 2: Generative AI Tasks

This repository contains the materials for Assignment 2, including three experiment notebooks and a small Flask web application for the image translation task. The goal is to show how different generative AI approaches were explored and how the best-performing demo was packaged for end users.

## Project Structure

### 1. Notebooks
- `Q1_CycleGAN_Image_Translation.ipynb`  
  This notebook focuses on the image translation task using a CycleGAN-style approach. It includes the setup for loading data, preprocessing images, defining the generator/discriminator logic, and experimenting with sketch-to-photo and photo-to-sketch conversion.

- `Q2_English_to_Urdu_Translation.ipynb`  
  This notebook covers the language translation task. It explains how the dataset is prepared, how text is tokenized, and how a sequence-to-sequence style model is trained or tested for English-to-Urdu translation.

- `Q3_Diffusion_UNet_Generation.ipynb`  
  This notebook explores diffusion-based generation using a U-Net architecture. It is intended to show how generative modeling can be approached using denoising steps and model-based sampling.

### 2. Flask Demo
- `flask_app/`  
  This folder contains the runnable web application for the image translation demo. It allows a user to upload an image and receive a translated output without needing to work directly inside the notebook.

- `flask_app/app.py`  
  This file loads the trained Keras models, handles image upload requests, preprocesses the input image, runs inference, and returns the translated result to the frontend.

- `flask_app/templates/index.html`  
  This is the webpage shown to the user. It provides the upload form, displays any error messages, and renders the generated output image.

### 3. Report and Visuals
- `GEN_AI_Assignment_2_Report.pdf`  
  This is the assignment report that explains the methodology, experiment setup, and results.

- Sample images in this folder (such as `photo_to_sketch_sample.png` and `sketch_to_photo_sample.png`) are included to visually demonstrate the expected input/output behavior.

## Features

- Upload an image through a simple web interface
- Automatically detect whether the input appears to be a photo or a sketch
- Generate a translated output using the trained model
- Display error messages clearly when validation fails
- Keep the notebook experiments and the runnable demo in one repository

## How the Project Works

1. The notebooks document the experiments, preprocessing steps, and model exploration.
2. The Flask demo turns the most practical result into an interactive browser-based tool.
3. The user uploads an image, the app preprocesses it, and the model generates the transformed output.
4. The result is displayed directly on the webpage for easy inspection.

## Setup Instructions

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the Flask application:

   ```bash
   cd flask_app
   python app.py
   ```

3. Open the local URL shown by Flask (usually `http://127.0.0.1:5000`) in your browser.

## Requirements

The project depends on Python libraries such as:
- Flask
- TensorFlow / Keras
- NumPy
- Pillow
- Matplotlib

A `requirements.txt` file is included to help install the main dependencies.

## Notes for GitHub Publishing

- Large model files and checkpoint artifacts should not be committed unless you are using Git LFS or another storage solution.
- The notebooks are kept here for documentation and reproducibility.
- The Flask demo is the most direct way to showcase the final result to a user.

