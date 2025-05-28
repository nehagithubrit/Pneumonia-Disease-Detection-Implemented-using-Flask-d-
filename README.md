🩺 PneumoScan
PneumoScan is a deep learning–powered system designed to detect pneumonia from chest X-ray images. It leverages convolutional neural networks (CNNs) to identify infection patterns and abnormalities in lungs, enabling early diagnosis and assisting healthcare professionals in efficient clinical decision-making.

This project combines medical image processing, deep learning models, and explainable AI techniques to improve diagnostic accuracy and interpret model predictions with transparency.

📌 Features
🧠 Pneumonia Detection Using CNNs: Automatically identifies pneumonia in chest X-ray images using deep learning models like VGG16, ResNet50, and custom CNN architectures.

⚖️ Class Imbalance Handling: Addresses skewed datasets using techniques like data augmentation, oversampling, and focal loss to improve performance on minority (pneumonia) cases.

🔍 Explainable Predictions: Utilizes Grad-CAM and heatmaps to visualize important regions in X-ray images that influence the model’s decision, enhancing model interpretability.

📊 Evaluation Metrics Optimized: Assesses model using accuracy, precision, recall, F1-score, and AUC-ROC to reflect real-world clinical relevance.

🧹 Robust Preprocessing Pipeline: Includes grayscale normalization, image resizing, noise reduction, and histogram equalization for optimal image quality.

📈 Model Comparison Dashboard: Offers visual comparisons of different models based on performance metrics, confusion matrix, and misclassified samples.

💻 User-Friendly Interface: (Optional) Integrated with a web dashboard Using Tailwind CSS and Flask app for uploading X-rays and receiving instant diagnostic results.

### 🛠 Tech Stack

- **Python** – Core programming language used for scripting and implementation.
- **NumPy, Pandas** – Used for data manipulation, cleaning, and preprocessing.
- **Scikit-learn** – Utilized for data splitting, model evaluation metrics, and baseline models.
- **XGBoost** – Applied as an advanced ensemble model for performance benchmarking.
- **Matplotlib, Seaborn** – Employed for creating insightful visualizations such as confusion matrix, ROC curves, and Grad-CAM overlays.
- **Streamlit / Flask** *(optional)* – Used to deploy the trained model with a user-friendly interface for real-time chest X-ray classification.

📸 Screenshot
![image](https://github.com/user-attachments/assets/f3c38d59-5310-4855-a328-68975a052043)

![image](https://github.com/user-attachments/assets/24fe36a4-924f-4f0b-909b-8a5f7531109e)
