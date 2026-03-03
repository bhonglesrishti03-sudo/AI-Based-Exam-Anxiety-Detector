## AI Exam Anxiety Detector

An end-to-end Natural Language Processing (NLP) system designed to detect exam-related anxiety from textual inputs using transformer-based deep learning models. The project focuses on identifying anxiety patterns from mental-health text data and provides a deployable demo interface for real-time predictions.

## Project Objectives

Analyze text-based mental-health data to identify anxiety-related patterns

Leverage transformer-based models (BERT) for accurate text classification

Build a clean and modular ML pipeline from data preprocessing to deployment

Provide a user-friendly frontend for real-time anxiety detection

## Environment Setup and Configuration
1) Python Installation and Virtual Environment

Python version 3.9 or above is used for this project.
A dedicated virtual environment is created to ensure isolated and clean dependency management.

python -m venv venv
venv\Scripts\activate

2) Library Installation

All required libraries for NLP processing, BERT model integration, backend services, and frontend UI are installed using a centralized requirements.txt file.

pip install -r requirements.txt

3) Environment Validation

The virtual environment is validated to ensure successful installation of key libraries including:

PyTorch

Hugging Face Transformers

FastAPI

Streamlit

This confirms readiness for model training, API development, and UI rendering.

## Project Directory Structure

A modular and scalable project structure is maintained to separate concerns clearly.

AI-EXAM-ANXIETY-DETECTOR/
│
├── backend/        # Backend API services
├── frontend/       # Streamlit-based user interface
├── src/            # Core ML logic (preprocessing, training, modeling)
├── data/           # Dataset directory (ignored in GitHub)
├── model/          # Trained model storage (ignored in GitHub)
│
├── requirements.txt
├── .gitignore
└── README.md

# GPU and Training Environment

Model training and experimentation are performed using Google Colab to leverage GPU acceleration.
The environment is verified for CUDA availability to support transformer-based model training.

## Version Control Setup

Git and GitHub are configured for version control and collaborative development.
Large files such as datasets and trained model weights are excluded using .gitignore to maintain repository hygiene.

## Dataset Selection and Organization
1) Dataset Collection

A publicly available mental-health text dataset was obtained from Kaggle. The dataset contains thousands of text statements annotated with emotional and psychological categories including:

Normal

Anxiety

Stress

Depression

Suicidal

Among the available dataset files, mental_health_combined.csv was selected as the primary dataset as it consolidates all categories into a single structured file suitable for transformer-based text classification.

Additional dataset files were used only for analytical understanding:

An unbalanced dataset to study class distribution

A feature-engineered dataset not required for BERT-based modeling

Note: Dataset files are excluded from this repository due to licensing and size constraints. Users must download the dataset separately from Kaggle and place the required CSV file inside the data/ directory.

2) DataSet Loading
To inspect the dataset, a Jupyter Notebook named

01_dataset_inspection.ipynb is created inside the notebooks/ folder.

The dataset is loaded using the Pandas library:

import pandas as pd

df = pd.read_csv("../data/final_anxiety_dataset.csv")

This allows structured analysis and easy visualization of dataset properties.

3) Understanding DataSet Structure
The dataset mainly consists of two important columns:

statement → Text input written by users

status → Mental health label associated with the text

To understand the dataset size and structure, the following commands are used:

df.head()

df.shape

df.columns

This step confirms:

Number of records

Number of columns

Type of data present.

4) Finding Missing Vaues
Before training any model, missing values must be identified.

df.isnull().sum()

This step reveals that some text entries may be missing and need to be handled during preprocessing.

5) Class Distribution Analysis
   The dataset contains multiple mental-health categories, which may not be evenly distributed.

df['status'].value_counts()

This analysis helps identify:

Class imbalance

Dominant mental-health categories

Need for label mapping in later stages

Understanding class distribution is essential to prevent model bias.

# Data Processing & Label Mapping
3.2 Original Dataset Labels

The original dataset consists of multiple mental-health categories representing different emotional and psychological states, including:

Normal

Anxiety

Stress

Depression

Suicidal

Bipolar

Personality Disorder

These labels reflect clinical or psychological conditions, whereas the goal of this project is to analyze exam-related anxiety levels, not to perform medical diagnosis.

3.3 Custom Label Mapping Strategy

A custom label-mapping strategy is designed based on emotional intensity and severity.

This mapping converts complex mental-health categories into a simpler, practical anxiety scale consisting of three levels:

Low Anxiety

Moderate Anxiety

High Anxiety

The mapping ensures that the classification output aligns with real-world academic stress scenarios while remaining ethically responsible and interpretable.

Label Mapping Table
Original Mental-Health Label	Anxiety Level
Normal	Low Anxiety
Stress	Moderate Anxiety
Anxiety	Moderate Anxiety
Depression	High Anxiety
Suicidal	High Anxiety
Bipolar	High Anxiety
Personality Disorder	High Anxiety
Rationale Behind the Mapping

Low Anxiety indicates emotionally stable or neutral states.

Moderate Anxiety captures common exam-related stress and nervousness.

High Anxiety represents severe emotional distress that can significantly affect academic performance.

This strategy reduces classification complexity while preserving emotional severity and relevance to exam anxiety detection.
