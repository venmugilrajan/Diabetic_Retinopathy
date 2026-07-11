# Live view:https://diaretino.vercel.app/
---
title: Diabetic Retinopathy Diagnostic System
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
python_version: "3.10"
---

# 👁️ Diabetic Retinopathy Classification System

An end-to-end deep learning pipeline to classify Diabetic Retinopathy severity from retinal fundus images using PyTorch, EfficientNet-B3, Focal Loss, and a premium Next.js dashboard + Gradio console.

## 📁 Project Structure
```
├── src/                              # Next.js App Router codebase
│   └── app/                          # Main pages, layouts, and API routes
├── diabetes-re/                      # Labeled Retina Dataset (Ignored in Spaces)
│   ├── train.csv                     # Image names & diagnoses
│   └── colored_images/               # Preprocessed image subfolders
├── code.ipynb                        # EDA, Preprocessing, Training & Evaluation Notebook
├── app.py                            # Gradio Web Server Dashboard interface (Space app file)
├── best_diabetic_retinopathy_model.pth # Saved model weights checkpoint
├── requirements.txt                  # Python dependencies list
├── package.json                      # Next.js configuration and dependencies
├── next.config.js                    # Next.js bundler settings
└── README.md                         # Project documentation
```

---

## 🎨 Premium Features

1. **Dual Premium Modes**: Elegant transition between dark cyber-obsidian and glassmorphic light theme presets.
2. **Clinician Annotation Canvas**: Real-time canvas sketching directly overlaying the fundus photograph to draw lesion boundaries.
3. **Graham Preprocessed View**: Gaussian blur subtraction technique sharpening vascular profiles side-by-side with original photos.
4. **AI Grad-CAM Heatmapping**: Neural network activation map highlighting visual points of interest.

---

## 🛠️ Setup & Installation

### 1. Python ML Backend
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the local backend server:
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:7860` to access the local Gradio interface.

### 2. Next.js Web Dashboard
1. Install node dependencies in the root folder:
   ```bash
   npm install
   ```
2. Launch the local development server:
   ```bash
   npm run dev
   ```
3. Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🚀 Hosting & Deployment

### 1. Hugging Face Spaces (Backend)
- This directory is configured with Gradio space frontmatter metadata. 
- Create a Gradio Space on Hugging Face and push `app.py`, `requirements.txt`, and `best_diabetic_retinopathy_model.pth` to run the model 24/7.

### 2. Vercel (Next.js Frontend)
1. Push your repository to GitHub.
2. Link your repository to a new project in [Vercel](https://vercel.com/).
3. Vercel will automatically build and deploy your Next.js application at a public link.
4. Click the gear icon (`⚙️`) in your website's header to save your Hugging Face Space path (e.g. `venmugilrajan/Diabetic_Retinopathy`) so your Vercel frontend automatically connects to the model in the cloud!
