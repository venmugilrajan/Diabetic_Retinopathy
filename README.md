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

An end-to-end deep learning pipeline to classify Diabetic Retinopathy severity from retinal fundus images using PyTorch, EfficientNet-B3, Focal Loss, and Gradio.

## 📁 Project Structure
```
├── diabetes-re\                      # Labeled Retina Dataset (Ignored in Spaces)
│   ├── train.csv                     # Image names & diagnoses
│   └── colored_images\               # Preprocessed image subfolders
│       ├── No_DR\                    # Severity 0
│       ├── Mild\                     # Severity 1
│       ├── Moderate\                 # Severity 2
│       ├── Severe\                   # Severity 3
│       └── Proliferate_DR\           # Severity 4
│
├── code.ipynb                        # EDA, Preprocessing, Training & Evaluation Notebook
├── app.py                            # Gradio Web Server Dashboard interface
├── best_diabetic_retinopathy_model.pth # Saved model weights checkpoint
├── requirements.txt                  # Python dependencies list
└── README.md                         # Project documentation
```

## 🛠️ Setup & Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Make sure your dataset is structured inside the `diabetes-re/colored_images/` directory.

## 🏋️ Training the Model
Open `code.ipynb` in VS Code or Jupyter, select your active kernel, and run the cells sequentially:
* **Preprocessing**: Applies Ben Graham's method (local contrast subtraction) to emphasize retinal lesions.
* **Loss Function**: Employs Focal Loss combined with Effective Class-Balanced Weights to resolve the high class-imbalance.
* **Architecture**: Fine-tunes a pre-trained EfficientNet-B3 backbone.

## 🌐 Launching the Web App
Run the local web server:
```bash
python app.py
```
Open the local URL in your browser: `http://127.0.0.1:7860` to upload scans and review diagnostic assessments.

---

## 🎨 Premium Next.js Frontend (Vercel Ready)

We have built a premium, modern React/Next.js dashboard inside the `frontend/` directory. This frontend features a futuristic retina diagnostic style, drag-and-drop file inputs, scanning animations, and detailed diagnostic graphs.

### Local Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
4. Open [http://localhost:3000](http://localhost:3000) in your browser.

### Connecting Frontend to Backend
- Click on the **Backend API** button at the top right of the dashboard.
- Enter your running backend's endpoint URL (e.g., your local `http://localhost:7860` or your Hugging Face Space URL like `https://username-space-name.hf.space`).
- Click **Save Changes**.

### 🚀 Hosting on Vercel
1. Push your repository to GitHub, GitLab, or Bitbucket.
2. Go to [Vercel](https://vercel.com/) and click **Add New Project**.
3. Select your repository.
4. Set the **Root Directory** of the project to `frontend`.
5. Click **Deploy**. Vercel will automatically build and publish your Next.js application.

