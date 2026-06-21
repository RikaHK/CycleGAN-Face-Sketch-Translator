# Assignment 2: Generative AI Tasks

This folder contains the three assignment notebooks and a small Flask demo that showcases the image translation workflow.

## Folder Overview

- `Q1_CycleGAN_Image_Translation.ipynb` - notebook for the image translation / sketch-photo task
- `Q2_English_to_Urdu_Translation.ipynb` - notebook for the English-to-Urdu translation task
- `Q3_Diffusion_UNet_Generation.ipynb` - notebook for the diffusion / U-Net generation task
- `flask_app/` - runnable web demo for the face sketch/photo translator
- `GEN_AI_Assignment_2_Report.pdf` - assignment report
- Sample images and model output visuals for reference

## Running the Flask Demo

```bash
cd flask_app
python app.py
```

Then open the local URL shown by Flask in your browser.

## Important Notes

- The notebooks are included for documentation and reference.
- The Flask demo is the most practical part to run directly.
- The demo expects the trained `.keras` model files to be present in the same folder.
- If you plan to publish this repository publicly, keep large model/checkpoint files out of the main commit unless you are using a proper storage solution.
