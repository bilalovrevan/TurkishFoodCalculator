# FoodVisionAI 🍽️  
AI-powered Turkish Food Recognition and Nutrition Estimation System

## Overview
FoodVisionAI is an end-to-end artificial intelligence system designed to recognize Turkish meals from images and estimate their nutritional values.  
The system combines deep learning-based image classification with a nutrition database to provide calorie and macronutrient information based on portion size.

This project was developed as part of an academic data science and artificial intelligence assignment and focuses on practical AI system design, deployment, and usability.

---

## Features
• Image-based Turkish food recognition  
• Deep learning model trained on multi-source datasets  
• Confidence-based food detection  
• Portion-based nutrition calculation  
• FastAPI backend for model inference  
• Interactive web-based user interface  
• Modular and extensible project structure  

---

## Project Architecture
The system consists of three main components:

1. **Model Layer**
   - ResNet18-based convolutional neural network
   - Trained on Food101 and Turkish food datasets
   - Outputs food class probabilities

2. **Backend Layer**
   - FastAPI-based REST API
   - Handles image uploads and predictions
   - Applies confidence thresholding
   - Calculates nutrition values dynamically

3. **Frontend Layer**
   - HTML, CSS, and JavaScript-based UI
   - Image upload and portion selection
   - Displays predictions and nutrition results

---

## Dataset
Multiple data sources were combined to improve coverage and accuracy:

• Food101 dataset  
• Turkish food datasets from Kaggle and HuggingFace  
• Additional images collected via web scraping  

All datasets were cleaned, unified, and standardized before training.

---

## Model
• Architecture: ResNet18  
• Framework: PyTorch  
• Input size: 224 × 224 RGB images  
• Output: Food class probabilities  
• Confidence threshold: 45%  

If the confidence score is below the threshold, the system returns "Not a food image".

---

## Nutrition Estimation
• Nutrition values are stored in a structured JSON database  
• All base values are defined for a standard 180g portion  
• User-selected portion size dynamically scales the values  
• A fallback estimation logic is used if a food is missing from the database  

Macronutrients provided:
• Calories  
• Protein  
• Fat  
• Carbohydrates  

---

## API Endpoints
### Health Check
GET /health

Returns system status and device information.

### Prediction


POST /predict

**Parameters**
• file: food image  
• grams: portion size  

**Response**
• food_detected  
• food_name  
• confidence  
• calories  
• protein  
• fat  
• carbs  

---

## Installation
Clone the repository:


git clone https://github.com/bilalovrevan/TurkishFoodCalculator.git

cd FoodVisionAI


Create and activate virtual environment:


python -m venv .venv
source .venv/bin/activate # Linux / macOS
.venv\Scripts\activate # Windows


Install dependencies:


pip install -r requirements.txt


---

## Running the Backend


uvicorn api:app --reload


Backend will be available at:


http://127.0.0.1:8000


---

## Running the Frontend
Open `index.html` in a browser or serve it via a local web server.

---

## Project Structure


FoodVisionAI/
│
├── api.py
├── requirements.txt
├── models/
│ ├── best_foodvision_model.pth
│ └── classes.txt
├── backend/
│ ├── nutrition.json
│ └── backend_data/
├── dataset/
├── unified_dataset/
├── train_model.py
├── train_clean.py
├── predict.py
├── index.html
├── css/
├── js/
└── README.md


---

## Limitations
• Limited performance on visually similar dishes  
• Nutrition values are approximations  
• Model performance depends on image quality  

---

## Future Improvements
• Larger and more diverse Turkish food dataset  
• Advanced architectures such as EfficientNet or ViT  
• Multi-food detection in a single image  
• User accounts and nutrition history tracking  
• Mobile application support  

---

## Conclusion
FoodVisionAI demonstrates a complete AI pipeline from dataset preparation and model training to deployment and user interaction.  
The project highlights practical decision-making in AI system design and provides a strong foundation for future research or product development.

---

## Author
Ravan Bilalov  
MSc Data Science & Artificial Intelligence  
SRH University, Germany
