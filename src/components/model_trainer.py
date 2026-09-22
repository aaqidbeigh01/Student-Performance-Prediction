import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            logging.info("Training Random Forest Regressor")
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            score = r2_score(y_test, predictions)
            logging.info(f"Model Training completed with R2 Score: {score}")

            if score < 0.5:
                raise CustomException("Model score is too low", sys)

            logging.info("Saving best model object")
            save_object(self.config.trained_model_file_path, model)
            return score
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    
    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion()
    
    transformation = DataTransformation()
    train_arr, test_arr = transformation.initiate_data_transformation(train_data, test_data)
    
    trainer = ModelTrainer()
    r2 = trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Pipeline complete. Check the logs/ folder for execution details. Model R2 Score: {r2}")