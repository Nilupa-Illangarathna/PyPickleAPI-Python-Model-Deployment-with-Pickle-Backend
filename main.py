from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import ast
import torch
import copy
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from recommendation_functions import RecommendationFunctions  # Import the class
from transformations import *  # Import the class

app = Flask(__name__)

# Load the model and dataset
with open('recommendation_model.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# Extract the loaded dataset and functions
loaded_dataframe = loaded_data['dataframe']
loaded_functions = loaded_data['functions']

# Example Usage
csv_structure = {
    'Hotel_Address': 'str',
    'Review_Date': 'str',
    'Average_Score': 'float',
    'Hotel_Name': 'str',
    'Reviewer_Nationality': 'str',
    'Negative_Review': 'str',
    'Review_Total_Negative_Word_Counts': 'int',
    'Positive_Review': 'str',
    'Review_Total_Positive_Word_Counts': 'int',
    'Reviewer_Score': 'float',
    'Total_Number_of_Reviews_Reviewer_Has_Given': 'int',
    'Total_Number_of_Reviews': 'int',
    'Tags': 'str',
    'days_since_review': 'str',
    'Additional_Number_of_Scoring': 'int',
    'lat': 'float',
    'lng': 'float'
}

csv_path = 'feedback_loop.csv'  # Change this path accordingly

# Create an instance of the RecommendationFunctions class
loaded_recommendation_system = RecommendationFunctions(loaded_dataframe, csv_structure, csv_path)

@app.route('/initial_recommendations', methods=['POST'])
def initial_recommendations():
    data = request.get_json()
    num_recommendations = data.get('num_recommendations', 1000)
    mobile_or_postman = data.get("onWhich", "postman")
    # print(mobile_or_postman)
    recommendations = loaded_functions['initial_recommendations'](num_recommendations)

    # Process recommendations before sending to the frontend
    recommendations = [process_recommendation(recommendation) for recommendation in recommendations]
    recommendations = remove_single_quotes(recommendations)
    recommendations = remove_double_quotes(recommendations)
    if mobile_or_postman != "postman":
        recommendations = process_recommendations(recommendations)
        recommendations = process_summaries_and_save_files(recommendations)
    #     process_summaries_and_save_files_for_flutter(recommendations)

    return jsonify(recommendations)

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    query_sentence = data['query_sentence']
    num_recommendations = data.get('num_recommendations', 1000)
    mobile_or_postman = data.get("onWhich", "postman")
    # print(mobile_or_postman)

    recommendations = loaded_functions['get_recommendations'](query_sentence, num_recommendations)

    # print(recommendations[0])

    # Process recommendations before sending to the frontend
    recommendations = [process_recommendation(recommendation) for recommendation in recommendations]
    recommendations = remove_single_quotes(recommendations)
    recommendations = remove_double_quotes(recommendations)
    if mobile_or_postman != "postman":
        recommendations = process_recommendations(recommendations)
        recommendations = process_summaries_and_save_files(recommendations)
        # process_summaries_and_save_files_for_flutter(recommendations)
    # Send the processed recommendations to the frontend


    return jsonify(recommendations)


@app.route('/filter_rows', methods=['POST'])
def filter_rows():
    data = request.get_json()
    column_name = data['column_name']
    column_value = data['column_value']

    filtered_df = loaded_functions['filter_rows'](column_name, column_value)

    return jsonify(filtered_df.to_dict(orient='records'))


@app.route('/get_item_by_rank', methods=['POST'])
def get_item_by_rank():
    data = request.get_json()
    recommendations = data['recommendations']
    desired_rank = data['desired_rank']

    item_by_rank_row = loaded_functions['get_item_by_rank_from_recommendations'](recommendations, desired_rank)

    # # Process recommendations before sending to the frontend
    # item_by_rank_row = [process_recommendation(recommendation) for recommendation in item_by_rank_row]
    item_by_rank_row = remove_single_quotes(item_by_rank_row)
    item_by_rank_row = remove_double_quotes(item_by_rank_row)

    return jsonify(item_by_rank_row)

@app.route('/get_attribute_keys', methods=['POST'])
def get_attribute_keys():
    data = request.get_json()
    recommendation = data['recommendation']

    attribute_keys = loaded_functions['get_attribute_keys'](recommendation)

    return jsonify(attribute_keys)

# @app.route('/get_attribute_value', methods=['POST'])
# def get_attribute_value():
#     data = request.get_json()
#     recommendation = data['recommendation']
#     key = data['key']
#
#     attribute_value = loaded_functions['get_attribute_value'](recommendation, key)
#
#     return jsonify(attribute_value)

@app.route('/get_attribute_value', methods=['POST'])
def get_attribute_value():
    data = request.get_json()
    recommendation = data.get('recommendation')
    key = data.get('key')

    if recommendation is None or key is None:
        return jsonify({"error": "Invalid request. 'recommendation' and 'key' are required parameters."}), 400

    attribute_value = loaded_functions['get_attribute_value'](recommendation, key)

    return jsonify({"attribute_value": attribute_value})


@app.route('/get_column_names_from_entire_row', methods=['POST'])
def get_column_names_from_entire_row():
    data = request.get_json()
    entire_row = data['entire_row']

    column_names = loaded_functions['get_column_names_from_entire_row'](entire_row)

    return jsonify(column_names)

@app.route('/get_column_value_from_entire_row', methods=['POST'])
def get_column_value_from_entire_row():
    data = request.get_json()
    entire_row = data['entire_row']
    column_name = data['column_name']

    column_value = loaded_functions['get_column_value_from_entire_row'](entire_row, column_name)

    return jsonify(column_value)

@app.route('/edit_column_value', methods=['POST'])
def edit_column_value():
    data = request.get_json()
    row = data['row']
    column_name = data['column_name']
    new_value = data['new_value']

    updated_row = loaded_functions['edit_column_value'](row, column_name, new_value)

    return jsonify(updated_row)

@app.route('/get_dataset_size', methods=['GET'])
def get_dataset_size():
    dataset_size = loaded_functions['get_dataset_size']()

    return jsonify({'dataset_size': dataset_size})

@app.route('/add_new_data_feedback_loop', methods=['POST'])
def add_new_data_feedback_loop():
    data = request.get_json()
    new_data = data['new_data']

    loaded_recommendation_system.add_new_hotel_data_feedback_loop(new_data)

    return jsonify({'message': 'New data added to the feedback loop.'})


# New route to read feedback loop CSV
@app.route('/read_feedback_loop_csv', methods=['GET'])
def read_feedback_loop_csv():
    num_rows = request.args.get('num_rows', default=5, type=int)

    read_rows = loaded_recommendation_system.read_feedback_loop_csv(num_rows)
    return jsonify({'read_rows': read_rows})


@app.route('/save_model', methods=['POST'])
def save_model():
    data = request.get_json()
    output_file_path = data['output_file_path']

    loaded_functions['save_model'](output_file_path)

    return jsonify({'message': 'Model and dataset saved successfully.'})

# Additional Endpoints for the New Functions
@app.route('/get_highest_ranked_recommendations', methods=['POST'])
def get_highest_ranked_recommendations():
    data = request.get_json()
    head_size = data['head_size']

    recommendations = loaded_functions['get_highest_ranked_recommendations'](head_size)

    # Process recommendations before sending to the frontend
    recommendations = [process_recommendation(recommendation) for recommendation in recommendations]
    recommendations = remove_single_quotes(recommendations)
    recommendations = remove_double_quotes(recommendations)

    return jsonify(recommendations)

@app.route('/add_to_previous_preferences', methods=['POST'])
def add_to_previous_preferences():
    data = request.get_json()
    sentence = data['sentence']

    loaded_functions['add_to_previous_preferences'](sentence)

    return jsonify({'message': 'Sentence added to previous preferences.'})

@app.route('/get_recommendations_from_previous', methods=['POST'])
def get_recommendations_from_previous():
    data = request.get_json()
    query_sentence = data['query_sentence']
    num_recommendations = data.get('num_recommendations', 5)

    recommendations = loaded_functions['get_recommendations_from_previous'](query_sentence, num_recommendations)

    # Process recommendations before sending to the frontend
    recommendations = [process_recommendation(recommendation) for recommendation in recommendations]
    recommendations = remove_single_quotes(recommendations)
    recommendations = remove_double_quotes(recommendations)

    return jsonify(recommendations)






if __name__ == '__main__':
    app.run(port=3500)