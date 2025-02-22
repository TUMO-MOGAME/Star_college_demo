import sys
from src.components.data_ingestion import dataIngestion
from src.components.data_validation import datavalidation
from src.components.data_transformation import dataTransformation
from src.components.model_training import modelTrainer

def main():
    try:
        # Step 1: Data Ingestion
        data_ingestion = dataIngestion()
        train_data_path, test_data_path = data_ingestion.initialize_dataIngestion()
        print(f"Data Ingestion completed.\nTrain data: {train_data_path}\nTest data: {test_data_path}")

        # Step 2: Data Validation
        data_validation = datavalidation("artifacts\\train_data.csv","artifacts\\test_data.csv")
        validation_status = data_validation.validate_data()

        print(f"Validation Status: {validation_status}")  # Add this line for debugging

        if not validation_status:
            print("Data Validation failed. Exiting...")
            sys.exit(1)

        print("Data Validation completed successfully.")


        # Step 3: Data Transformation
        data_transformation = dataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.start_data_transformation("artifacts\\train_data.csv","artifacts\\test_data.csv")
        print(f"Data Transformation completed. Preprocessor saved at: {preprocessor_path}")

        # Step 4: Model Training
        model_trainer = modelTrainer()
        model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(f"Model Training completed. Model Score: {model_score}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()