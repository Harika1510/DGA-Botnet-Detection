# DGA-Botnet-Detection
Detection of DGA Botnets using Explainable AI
Detecting DGA (Domain Generation Algorithm) Botnets using XGBoost with Explainable AI (SHAP) for better transparency and accuracy.

## Abstract
Cyberattacks using DGA Botnets are a major threat to cybersecurity. Traditional detection methods are often rigid and lack real-time transparency.
This project proposes a machine learning-based solution using XGBoost and SHAP (SHapley Additive Explanations) to accurately detect DGA botnet traffic with enhanced interpretability.

## Dataset
Name: Botnet DGA Dataset
Source: Originally from Kaggle.
Sample Used: BotnetDgaDataset_sample.csv (100 rows)
Note: Full dataset can be downloaded from Kaggle (or available upon request).

## Technologies Used
Python 3.11
XGBoost
SHAP (Explainable AI)
Scikit-Learn
Pandas, NumPy
Matplotlib, Seaborn
Tkinter (for simple UI)

## Machine Learning Approach
Stage	Details
Feature Extraction	Domain name entropy, character frequency, domain length, reputation metrics
Model	XGBoost Classifier
Explainability	SHAP Values to interpret model predictions

## Performance Metrics
Accuracy
Precision
Recall
F1-Score
Confusion Matrix

✅ High detection accuracy
✅ Reduced false positives
✅ Real-time adaptability

## Key Advantages
Explainable AI (XAI): Helps cybersecurity professionals understand model decisions.
High Detection Accuracy: Outperforms traditional methods.
Real-Time Threat Detection: Faster response to botnet attacks.
Scalability: Handles large volumes of network traffic.

## Future Scope
Integrate real-time live network traffic analysis.
Use advanced deep learning models like Transformers.
Deploy decentralized detection systems using federated learning.
Collaborate with industry experts for building stronger AI-driven security solutions.

 Thank you for checking out the project! 
