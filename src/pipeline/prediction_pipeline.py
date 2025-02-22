import sys
import os
import pandas as pd
from src.exception import ProjectException  # Custom exception for consistent error handling
from src.utils import load_object         # Utility function to load serialized objects (model, preprocessor)

# -----------------------------------------------------------------------------
# PredictPipeline Class
# -----------------------------------------------------------------------------
class PredictPipeline:
    def __init__(self):
        # The constructor does not perform any initialization here.
        # In a production setup, you might load the model and preprocessor once here.
        pass

    def predict(self, features):
        """
        This method loads the saved model and preprocessor objects,
        applies the preprocessor to the incoming features, and then uses
        the model to predict the output.

        :param features: A pandas DataFrame containing the input features.
        :return: The predictions made by the model.
        """
        try:
            # Define the paths to the serialized model and preprocessor files
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            # Debug messages to track the loading process
            print("Before Loading")

            # Load the model and preprocessor objects using the load_object utility function
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            # Transform the input features using the preprocessor (ensures same format as training)
            data_scaled = preprocessor.transform(features)

            # Generate predictions using the loaded model
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            # If any error occurs, wrap it in a CustomException for consistent error handling
            raise ProjectException(e, sys)


# -----------------------------------------------------------------------------
# CustomData Class
# -----------------------------------------------------------------------------
class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education,  # Could also specify type e.g., str if known
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        """
        The CustomData class is designed to capture the raw input features for prediction.
        It stores each input as an instance variable.
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        This method converts the stored input features into a pandas DataFrame.
        The DataFrame format is required by the scikit-learn preprocessor and model.

        :return: A pandas DataFrame constructed from the input data.
        """
        try:
            # Create a dictionary where each key is a feature name and the value is a list containing the feature value.
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert the dictionary into a pandas DataFrame and return it.
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            # Wrap any exception in a CustomException to maintain consistent error handling
            raise ProjectException(e, sys)
