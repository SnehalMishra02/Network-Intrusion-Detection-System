# Network Intrusion Detection System

This project implements a **Network Intrusion Detection System** using a batch-trained XGBoost model and supports online learning with a Hoeffding Tree. The system is designed to classify network traffic, detect data drift, and evaluate model performance.

## Project Overview

The application allows users to:
- Upload a CSV file containing network traffic data.
- Detect data drift in the uploaded data compared to a reference dataset.
- Classify network traffic using a pre-trained XGBoost model.
- Perform online learning using a Hoeffding Tree model.
- Evaluate the performance of both batch and online models.
- Download the prediction results for further analysis.

## Features

1. **Batch Model**: A pre-trained XGBoost model is used for initial classification.
2. **Online Learning**: A Hoeffding Tree model adapts to new data in real-time.
3. **Drift Detection**: Detects significant changes in feature distributions using the Kolmogorov-Smirnov test.
4. **Performance Metrics**: Displays accuracy metrics for batch, online, and hybrid models.
5. **Downloadable Results**: Allows users to download predictions in CSV format.

## Technologies Used

- **Python**: Programming language.
- **Streamlit**: Web application framework for building interactive dashboards.
- **XGBoost**: Gradient boosting library for batch classification.
- **River**: Library for online machine learning.
- **Scikit-learn**: For preprocessing and evaluation metrics.
- **Pandas**: For data manipulation.
- **NumPy**: For numerical computations.

## How to Run the Application

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>

2. Install the required dependencies

3. Run the Streamlit application:
    ```bash
   streamlit run app.py


5. Open the application in your browser at 
http://localhost:8501.

##Project Contributors
This project was completed as part of the Network Information and Security course (Course ID: BITE401L) during the Winter Semester 2024-25 at Vellore Institute of Technology.

##Team Members:
Snehal Mishra (22BIT0325)
Priyanshi Saraf (22BIT0649)

##Project Guide:
Dr. Aswani Kumar Cherukuri

##Acknowledgments
We would like to express our gratitude to Dr. Aswani Kumar Cherukuri for his guidance and support throughout the project. This project was a valuable learning experience and helped us gain practical insights into network intrusion detection and machine learning.

##License
This project is for educational purposes and is not intended for commercial use.
