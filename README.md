# Student Performance Predictor 🎓

I built this machine learning pipeline to predict a student's math score based on real-world factors like parental education, lunch plans, and test prep history. 

The main goal of this project wasn't just to train a model, but to write clean, modular code. Instead of leaving everything tangled in a single Jupyter Notebook, I structured this as an end-to-end pipeline (inspired by Krish Naik's methodologies) with custom logging, exception handling, and a web interface.

## How It Works Under the Hood

*   **Data Ingestion:** Pulls in the raw data and cleanly splits it into training and testing sets.
*   **Data Transformation:** Handles the messy stuff. It imputes missing values, applies One Hot Encoding to categorical data (like gender and ethnicity), and scales the numbers. The whole process is saved as `preprocessor.pkl`.
*   **Model Training:** Tests out a few different algorithms. Random Forest performed the best here (hitting roughly an 85% $R^2$ score), so the pipeline automatically saves it as `model.pkl`.
*   **Flask Web App:** A lightweight frontend that connects to the prediction pipeline so you can actually test the model in your browser.

## Tech Stack
*   **Python 3.10**
*   **Machine Learning:** Scikit-Learn, Pandas, NumPy, XGBoost
*   **Web/Deployment:** Flask, HTML
*   **Tools:** Dill (for pickling), Git

## Folder Structure

```text
student_performance/
├── artifacts/              
├── logs/                   
├── notebooks/
│   └── data/               
├── src/
│   ├── components/         
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/           
│   │   └── predict_pipeline.py
│   ├── exception.py        # Error tracking to catch exact line numbers
│   ├── logger.py           # Logging
│   └── utils.py            
├── templates/              # HTML files for the web app
├── app.py                  # The Flask server
├── requirements.txt        
└── setup.py                # Makes the src folder installable as a package
