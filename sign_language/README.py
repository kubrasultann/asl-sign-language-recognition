Project Description

This project presents a real-time American Sign Language (ASL) letter recognition system based on hand landmark detection and machine learning techniques. The system recognizes static ASL alphabet gestures (excluding dynamic letters such as J and Z) and converts them into textual output.

Hand landmarks are extracted using MediaPipe Hand Landmarker, and classification is performed using a Random Forest model trained on a custom-collected dataset. The application supports both manual and automatic input modes, as well as word completion to improve usability.

The project is designed as an end-to-end pipeline including data collection, dataset analysis, model training, and real-time prediction.


Technologies Used

Programming Language: Python
Computer Vision: OpenCV
Hand Landmark Detection: MediaPipe Tasks (Hand Landmarker)
Machine Learning Model: Random Forest Classifier (scikit-learn)
Data Handling: NumPy, Pandas
Model Serialization: joblib


Project Structure

sign_language/
│
├─ models/
│   └─ hand_landmarker.task        # MediaPipe hand landmark model
│
├─ asl_text_app.py                 # Main application (final system)
├─ collect_asl_data.py             # Dataset collection script
├─ train_asl_model.py              # Model training script
├─ predict_asl_live.py             # Live prediction demo
├─ analyze_asl_dataset.py          # Dataset analysis script
├─ dataset_asl.csv                 # Collected ASL dataset
├─ asl_model.pkl                   # Trained Random Forest model
├─ words_en.txt                    # Word list for auto-completion
│
└─ user_samples/
   └─ user_manual_samples.csv      # Manually confirmed samples (not used in training)


Dataset Collection

The dataset was collected using the collect_asl_data.py script.
Each ASL letter sample consists of 63 numerical features representing the (x, y, z) coordinates of 21 hand landmarks.

To ensure a balanced dataset:

-The number of frames per class was monitored.
-Multiple recording sessions were performed.
-Dataset distribution was analyzed using analyze_asl_dataset.py.


Model Training

The model was trained using train_asl_model.py with the following steps:

1-Wrist-centered normalization of hand landmarks
2-Scale normalization using the middle finger MCP joint
3-Train-test split with stratification
4-Random Forest training with multiple decision trees

The trained model is saved as asl_model.pkl.


Real-Time Recognition

The real-time system is implemented in asl_text_app.py.
Main features include:
-Real-time ASL letter recognition
-Prediction stabilization using majority voting
-Manual and automatic input modes
-Word completion based on typed prefixes
-Visual user interface displaying predictions and confidence information

User Manual Samples (user_samples Folder)

The user_samples/user_manual_samples.csv file contains manually confirmed samples recorded when the user presses the ENTER key during manual mode.

Note:
These samples are not used in model training and do not affect prediction performance.
They are stored separately to demonstrate a potential future extension where user-confirmed samples could be incorporated into incremental or personalized training.

How to Run the Project
1. Collect Dataset (optional)
python sign_language/collect_asl_data.py

2. Train the Model
python sign_language/train_asl_model.py

3. Run Live Prediction
python sign_language/asl_text_app.py

Notes:

-Dynamic ASL letters (J, Z) are not included.
-The system focuses on static hand gestures.
-High classification accuracy is achieved due to controlled data collection and normalized landmark features.