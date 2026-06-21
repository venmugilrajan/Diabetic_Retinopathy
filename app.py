import os
import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import torch.nn as nn
import numpy as np
import cv2

class DiabeticRetinopathyModel(nn.Module):
    def __init__(self, num_classes=5, model_name='efficientnet_b3', pretrained=False):
        super(DiabeticRetinopathyModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'best_diabetic_retinopathy_model.pth'

model = DiabeticRetinopathyModel(num_classes=5, model_name='efficientnet_b3')
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Loaded weights successfully from {MODEL_PATH}")
else:
    print(f"Warning: Checkpoint {MODEL_PATH} not found.")
model = model.to(DEVICE)
model.eval()

class_names = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

clinical_explanations = {
    'No_DR': (
        "🔬 **No Diabetic Retinopathy Detected**\n\n"
        "**Clinical signs:** The retina appears healthy with no visible signs of damage.\n"
        "* No microaneurysms, hemorrhages, or exudates are detected.\n"
        "* Retinal blood vessels look uniform and intact.\n"
        "* Normal macula and optic disc appearance."
    ),
    'Mild': (
        "🔬 **Mild Non-Proliferative Diabetic Retinopathy (NPDR)**\n\n"
        "**Clinical signs:** This is the earliest stage of the disease.\n"
        "* Tiny balloon-like swellings in the retina's tiny blood vessels, called **microaneurysms**, are starting to form.\n"
        "* These microaneurysms may leak small amounts of fluid into the retina, but vision is usually not yet affected."
    ),
    'Moderate': (
        "🔬 **Moderate Non-Proliferative Diabetic Retinopathy (NPDR)**\n\n"
        "**Clinical signs:** The disease is progressing, and blood vessels show clear signs of blockage.\n"
        "* Presence of **intraretinal hemorrhages** (leaking blood dots) and **hard exudates** (yellowish lipid deposits).\n"
        "* Blocked vessels may cause **cotton wool spots** (small, fluffy white patches of nerve fiber swelling).\n"
        "* The retina begins to lose its proper blood supply."
    ),
    'Severe': (
        "🔬 **Severe Non-Proliferative Diabetic Retinopathy (NPDR)**\n\n"
        "**Clinical signs:** Many more blood vessels are blocked, cutting off blood supply to major parts of the retina.\n"
        "* Severe intraretinal hemorrhages present in all four quadrants of the eye.\n"
        "* Significant **venous beading** (blood vessels appearing like strings of beads).\n"
        "* High risk of progressing to the proliferative stage. Close clinical monitoring is required."
    ),
    'Proliferate_DR': (
        "🔬 **Proliferative Diabetic Retinopathy (PDR)**\n\n"
        "**Clinical signs:** This is the most advanced and dangerous stage of the disease.\n"
        "* The retina triggers the growth of new, abnormal, and fragile blood vessels (a process called **neovascularization**).\n"
        "* These new vessels can leak blood directly into the center of the eye (vitreous hemorrhage), blocking vision.\n"
        "* Without immediate medical intervention, this can lead to permanent vision loss or retinal detachment."
    )
}

class BenGrahamPreprocessing(object):
    def __init__(self, size=300, sigmaX=10):
        self.size = size
        self.sigmaX = sigmaX

    def __call__(self, img):
        img = np.array(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > 10
        if mask.any():
            img = img[np.ix_(mask.any(1), mask.any(0))]
        img = cv2.resize(img, (self.size, self.size))
        blurred = cv2.GaussianBlur(img, (0, 0), self.sigmaX)
        enhanced = cv2.addWeighted(img, 4, blurred, -4, 128)
        return Image.fromarray(enhanced)

val_transform = transforms.Compose([
    BenGrahamPreprocessing(size=300),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image):
    if image is None:
        return {}, "Please upload an image."
    
    img = Image.fromarray(image).convert('RGB')
    img_t = val_transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    prob_dict = {class_names[i]: float(probabilities[i]) for i in range(5)}
    predicted_idx = int(torch.argmax(probabilities).cpu().numpy())
    predicted_class = class_names[predicted_idx]
    explanation = clinical_explanations.get(predicted_class, "No explanation available.")
    
    return prob_dict, explanation

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 👁️ Diabetic Retinopathy Diagnostic System")
    gr.Markdown("Upload a retinal fundus scan to analyze and classify the severity level of Diabetic Retinopathy.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Retina Scan")
            submit_btn = gr.Button("Analyze Image", variant="primary")
            
        with gr.Column(scale=1):
            output_chart = gr.Label(num_top_classes=5, label="Severity Predictions")
            output_reason = gr.Markdown(label="Diagnostic Assessment")
            
    submit_btn.click(
        fn=predict, 
        inputs=input_image, 
        outputs=[output_chart, output_reason]
    )

if __name__ == "__main__":
    demo.launch(share=False)
