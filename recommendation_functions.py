import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import ast
import torch
import copy
import pickle
from datetime import datetime
import datetime as dt
import random
from typing import List, Dict


class RecommendationFunctions:
    def __init__(self, dataframe, csv_structure, csv_path):

        # current dataset storing related
        self.dataframe = dataframe
        # previous preferences storing related
        self.previous_preferences = []
        # new reviews preferences storing related
        # Create a DataFrame template with the provided CSV structure
        template_df = pd.DataFrame(columns=csv_structure.keys())

        # Save the template DataFrame as a CSV file
        template_df.to_csv(csv_path, index=False)

        self.csv_path = csv_path

    def get_highest_ranked_recommendations(self, head_size):  # Recommandations
        rank = 0
        """
        Get the highest-ranked set of recommendations based on the rows with the same month as the current date.

        Parameters:
        - head_size (int): The number of rows to retrieve.

        Returns:
        - highest_ranked_recommendations (pd.DataFrame): A DataFrame with highest-ranked hotel recommendations.
        """
        # Convert 'Review_Date_values' to datetime format
        self.dataframe['Review_Date_values'] = pd.to_datetime(self.dataframe['Review_Date_values'])

        # Get the current month
        current_month = dt.datetime.now().month

        # Filter rows with the same month as the current date
        current_month_data = self.dataframe[self.dataframe['Review_Date_values'].dt.month == current_month]

        # Assuming 'Reviewer_Score' is the column representing rankings
        highest_ranked_recommendations = current_month_data.sort_values(by='Average_Score_values',
                                                                        ascending=False).head(head_size)

        output_list = []
        for idx, row in highest_ranked_recommendations.iterrows():
            output_list.append({
                'rank': rank + 1,
                'similarity_score': -1,
                'entire_row': self.dataframe.loc[idx].to_dict()
            })
            rank += 1

        return output_list

    # Function 01: Add Data to Previous Preferences
    def add_to_previous_preferences(self, sentence):
        formatted_entry = {
            'sentence': sentence,
            'datetimerecorded': datetime.now()
        }

        self.previous_preferences.append(formatted_entry)
        print(self.previous_preferences)

    # Function 02: Get Recommendations Based on Previous Preferences
    def get_recommendations_from_previous(self, query_sentence, num_recommendations=5):  # Recommandations
        # Call get_recommendations method
        return self.get_recommendations(query_sentence, num_recommendations, use_previous=True)

    # Function 01: Get Recommendations
    def get_recommendations(self, query_sentence, num_recommendations=5, use_previous=False):  # Recommandations
        rank = 0
        model = SentenceTransformer('jinaai/jina-embedding-t-en-v1')

        if use_previous:
            # Use the previous preferences for recommendations
            if not self.previous_preferences:
                print("No previous preferences available.")
                return []

            # Combine previous preferences into a single string
            previous_preferences_text = ' '.join(entry['sentence'] for entry in self.previous_preferences)
            query_sentence = previous_preferences_text + ' ' + query_sentence

        try:
            self.dataframe['Positive_Review_Embeddings'] = self.dataframe['Positive_Review_Embeddings'].apply(
                ast.literal_eval)
        except Exception as e:
            problematic_rows = \
            self.dataframe[self.dataframe['Positive_Review_Embeddings'].apply(lambda x: not isinstance(x, dict))][
                'Positive_Review_Embeddings']

        # Use the entire dataset for recommendations
        positive_reviews_data = self.dataframe[
            ['Positive_Review', 'Positive_Review_Embeddings', 'Cleaned_Positive_Summmary']]

        positive_reviews_data = positive_reviews_data.dropna(subset=['Positive_Review_Embeddings'])

        positive_embeddings = positive_reviews_data['Positive_Review_Embeddings'].apply(
            lambda x: x[0]['embedding']).tolist()
        positive_sentences = positive_reviews_data['Positive_Review'].tolist()
        cleaned_positive_sentences = positive_reviews_data['Cleaned_Positive_Summmary'].tolist()

        if not positive_embeddings:
            print("No positive embeddings available.")
            return []

        query_embedding = model.encode([query_sentence], convert_to_tensor=True)

        # Debugging: Print the shapes of the input matrices
        print("Query embedding shape:", query_embedding.shape)
        print("Positive embeddings shape:", torch.tensor(positive_embeddings).shape)

        similarities = util.pytorch_cos_sim(query_embedding, torch.tensor(positive_embeddings))[0]

        results_df = pd.DataFrame({
            'Review': positive_sentences,
            'Cleaned_Positive_Summmary': cleaned_positive_sentences,
            'Positive_Review_Embeddings': positive_embeddings,
            'Similarity': similarities.tolist()
        })

        ranked_results = results_df.sort_values(by='Similarity', ascending=False).head(num_recommendations)

        output_list = []
        for idx, row in ranked_results.iterrows():
            output_list.append({
                'rank': rank + 1,
                'similarity_score': row['Similarity'],
                'entire_row': self.dataframe.loc[idx].to_dict()
            })
            rank += 1

        return output_list

    # Function 04: Initial Recommendations
    def initial_recommendations(self, num_recommendations=5):  # Recommandations
        # Check if there are entries in the previous preferences list
        if self.previous_preferences:
            # Get half of the recommendations from previous preferences
            num_previous_recommendations = num_recommendations // 2
            previous_recommendations = self.get_recommendations_from_previous('', num_previous_recommendations)
            print(f"Using {len(previous_recommendations)} recommendations from previous preferences.")
        else:
            previous_recommendations = []
            print("No previous preferences available.")

        # Get the remaining recommendations from the highest-ranked entries in the dataset
        num_remaining_recommendations = num_recommendations - len(previous_recommendations)
        highest_ranked_recommendations = self.get_highest_ranked_recommendations(num_remaining_recommendations)
        print(f"Using {len(highest_ranked_recommendations)} recommendations from the highest-ranked entries.")

        # Combine the recommendations
        combined_recommendations = previous_recommendations + highest_ranked_recommendations

        return combined_recommendations

    # Function 02: Filter Rows
    def filter_rows(self, column_name, column_value):
        filtered_df = self.dataframe[(self.dataframe[column_name]) == column_value]
        return filtered_df

    # Function 03: Get Item by Rank from Recommendations
    def get_item_by_rank_from_recommendations(self, recommendations, desired_rank):
        if 1 <= desired_rank <= len(recommendations):
            item = recommendations[desired_rank - 1]
            return item
        else:
            print("Invalid desired rank. Please provide a rank within the valid range.")
            return None

    # Function 04: Get Attribute Keys
    def get_attribute_keys(self, recommendation):
        return list(recommendation.keys())

    # Function 05: Get Attribute Value
    def get_attribute_value(self, recommendation, key):
        return recommendation.get(key)

    # Function 06: Get Column Names from Entire Row
    def get_column_names_from_entire_row(self, entire_row):
        return list(entire_row.keys())

    # Function 07: Get Column Value from Entire Row
    def get_column_value_from_entire_row(self, entire_row, column_name):
        return entire_row.get(column_name)

    # Function 08: Edit Column Value
    def edit_column_value(self, row, column_name, new_value):
        updated_row = copy.deepcopy(row)
        entire_row = updated_row.get("entire_row", {})
        entire_row[column_name] = new_value
        updated_row["entire_row"] = entire_row
        return updated_row

    # Function 09: Get Dataset Size
    def get_dataset_size(self):
        return len(self.dataframe)

    # Function 10: Add New Hotel Data Feedback Loop
    def add_new_hotel_data_feedback_loop(self, new_data, ):  # Feedback Loop
        # Read the existing feedback loop CSV file
        try:
            existing_df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print("Feedback loop CSV file not found.")
            return

        # Append the new data to the existing DataFrame
        new_df = pd.DataFrame(new_data, index=[0])
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)

        # Save the updated DataFrame back to the CSV file
        updated_df.to_csv(self.csv_path, index=False)

        # Update the current DataFrame in the class
        self.dataframe = updated_df

    def read_feedback_loop_csv(self, num_rows):  # Feedback Loop
        # Read the existing feedback loop CSV file
        try:
            existing_df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print("Feedback loop CSV file not found.")
            return pd.DataFrame()

        # Check if the requested number of rows is greater than available
        if num_rows >= len(existing_df):
            return existing_df.sample(frac=1).to_dict(orient='records')

        # Randomly select a specified number of unique rows
        selected_rows = existing_df.sample(n=num_rows, replace=False)
        return selected_rows.to_dict(orient='records')

    # Function 11: Save Recommendation Model
    def save_model(self, output_file_path):
        """
        Save the model and dataset using pickle.

        Parameters:
        - output_file_path (str): The file path to save the model and dataset.
        """
        with open(output_file_path, 'wb') as file:
            pickle.dump({
                'dataframe': self.dataframe,
                'functions': {
                    'get_highest_ranked_recommendations': self.get_highest_ranked_recommendations,
                    'add_to_previous_preferences': self.add_to_previous_preferences,
                    'get_recommendations_from_previous': self.get_recommendations_from_previous,
                    'get_recommendations': self.get_recommendations,
                    'initial_recommendations': self.initial_recommendations,
                    'filter_rows': self.filter_rows,
                    'get_item_by_rank_from_recommendations': self.get_item_by_rank_from_recommendations,
                    'get_attribute_keys': self.get_attribute_keys,
                    'get_attribute_value': self.get_attribute_value,
                    'get_column_names_from_entire_row': self.get_column_names_from_entire_row,
                    'get_column_value_from_entire_row': self.get_column_value_from_entire_row,
                    'edit_column_value': self.edit_column_value,
                    'get_dataset_size': self.get_dataset_size,
                    'add_new_hotel_data_feedback_loop': self.add_new_hotel_data_feedback_loop,
                    'read_feedback_loop_csv': self.read_feedback_loop_csv,
                    'save_model': self.save_model
                }
            }, file)
        print(f"Model and dataset saved to: {output_file_path}")

