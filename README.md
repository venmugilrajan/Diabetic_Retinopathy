# 👁️ Diabetic Retinopathy Classification System

An end-to-end deep learning pipeline to classify Diabetic Retinopathy severity from retinal fundus images using PyTorch, EfficientNet-B3, Focal Loss, and Gradio.

## 📁 Project Structure
* `code.ipynb` - Jupyter notebook containing EDA, data pipeline, training, and evaluation.
* `app.py` - Gradio web interface serving the trained model.
* `best_diabetic_retinopathy_model.pth` - Saved model weights checkpoint.
* `requirements.txt` - Python package dependencies list.

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
Open the local URL in your browser: `http://127.0.0.1:7861` to upload scans and review diagnostic assessments.
