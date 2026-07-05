# 🏦 Automated Explainable AI System for Loan Default Prediction

## 📝 Overview
This repository contains an end-to-end Automated Explainable AI (XAI) system designed to predict loan defaults. Beyond just providing a prediction, this project emphasizes **Explainability**, helping users and financial institutions understand the underlying factors and feature importance driving the AI's decision. 

## 🔗 Links & Resources
* **Live Web Application:** [Loan Default XAI Website](https://depi-loan-default-xai-frontend.onrender.com/)
* **Dataset:** [Loan Default Dataset on Kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default?hl=en-US)

## 🗂️ Repository Structure
* `Models/`: Serialized pre-trained machine learning models (e.g., `.pkl` files) ready for inference and deployment.
* `Website/`: Frontend source code (HTML, CSS, JavaScript) for the interactive web application interface.
* `data/`: Contains the datasets used for training and evaluating the models.
* `images/`: Stores visual assets, exploratory data analysis (EDA) plots, and Explainable AI visualizations (e.g., SHAP/LIME charts).
* `notebook/`: Jupyter Notebooks (`.ipynb`) detailing the data preprocessing pipeline, model training, and XAI implementation.
* `requirements.txt`: The list of Python dependencies needed to run the environment.

## 🛠️ Tech Stack
* **Languages:** Python, JavaScript, HTML, CSS
* **Environment:** Jupyter Notebook
* **Machine Learning:** `scikit-learn`, `pandas`, `numpy`
* **Explainable AI (XAI):** `shap`, `lime` 
* **Web Deployment:** Render

## ⚙️ Installation & Setup
To run this project or the notebooks locally, clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone [https://github.com/karimabdelmonem/automated-explainable-ai-system.git](https://github.com/karimabdelmonem/automated-explainable-ai-system.git)
cd automated-explainable-ai-system

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
