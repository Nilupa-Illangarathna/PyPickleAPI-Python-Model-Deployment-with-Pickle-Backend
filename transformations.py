import copy
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')



def remove_single_quotes(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = remove_single_quotes(value)
    elif isinstance(obj, list):
        obj = [remove_single_quotes(item) for item in obj]
    elif isinstance(obj, str):
        obj = obj.replace("'", "")
    return obj

def remove_double_quotes(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = remove_double_quotes(value)
    elif isinstance(obj, list):
        obj = [remove_double_quotes(item) for item in obj]
    elif isinstance(obj, str):
        obj = obj.replace("\"", "")
    return obj


def process_positive_review_dict(positive_review_dict):
    # Add double quotes to keys
    processed_dict = {f'"{key}"': value for key, value in positive_review_dict.items()}

    # Replace \" with "
    processed_dict = {key: value.replace('\\"', '"') if isinstance(value, str) else value for key, value in
                      processed_dict.items()}

    # Add quotes to string values
    processed_dict = {key: f'"{value}"' if isinstance(value, str) else value for key, value in processed_dict.items()}

    return processed_dict


def process_recommendation(recommendation):
    # Check if "Positive_Review_Dict" key exists
    if 'Positive_Review_Dict' in recommendation:
        # Process each positive review dictionary
        recommendation['Positive_Review_Dict'] = [
            process_positive_review_dict(positive_review_dict) for positive_review_dict in
            recommendation['Positive_Review_Dict']
        ]
    return recommendation

# ///////////////////////////////////







def process_recommendations(recommendations):
    modified_recommendations = []

    for recommendation in recommendations:
        if "entire_row" in recommendation:
            # Get the "entire_row" value
            entire_row = recommendation["entire_row"]

            # Remove the specified key-value pairs
            keys_to_remove = [
                "Positive_Review",
                "Positive_Review_Dict",
                "Positive_Review_Embeddings",
                #
                "Cleaned_Negative_Summmary",
                #
                "Cleaned_Positive_Summmary",
                #
                "Negative_sentence_column_Positivity_Percentage",
                "Negative_sentence_column_Negativity_Percentage",
                "Positive_sentence_column_Positivity Percentage",
                "Positive_sentence_column_Negativity Percentage",
                "Review_Date_values",
                # "Sequence",
                "Sequence_Number"
            ]
            for key in keys_to_remove:
                entire_row.pop(key, None)

            # Create a new recommendation item
            new_recommendation = {"entire_row": entire_row, "rank": recommendation["rank"], "similarity_score": recommendation["similarity_score"]}
            modified_recommendations.append(new_recommendation)

    return modified_recommendations


def process_summaries_and_save_files(recommendations):
    positive_filename = "positive_summaries.txt"
    negative_filename = "negative_summaries.txt"

    positive_file = open(positive_filename, "w", encoding="utf-8")
    negative_file = open(negative_filename, "w", encoding="utf-8")

    modified_recommendations = []

    for recommendation in recommendations:
        if "entire_row" in recommendation:
            entire_row = recommendation["entire_row"].copy()

            # Process Cleaned_Positive_Summmary
            if "Cleaned_Positive_Summmary" in entire_row:
                positive_summary = entire_row["Cleaned_Positive_Summmary"]
                sentences = sent_tokenize(positive_summary)
                if len(sentences) > 0:
                    entire_row["Cleaned_Positive_Summmary"] = sentences[0]
                    positive_file.write(sentences[0] + "\n")

            # Process Cleaned_Negative_Summmary
            if "Cleaned_Negative_Summmary" in entire_row:
                negative_summary = entire_row["Cleaned_Negative_Summmary"]
                sentences = sent_tokenize(negative_summary)
                if len(sentences) > 0:
                    entire_row["Cleaned_Negative_Summmary"] = sentences[0]
                    negative_file.write(sentences[0] + "\n")

            modified_recommendations.append({"entire_row": entire_row, "rank": recommendation["rank"], "similarity_score": recommendation["similarity_score"]})

    positive_file.close()
    negative_file.close()

    return modified_recommendations

def process_summaries_and_save_files_for_flutter(recommendations):
    positive_filename = "positive_summaries_flutter.txt"
    negative_filename = "negative_summaries_flutter.txt"

    positive_file = open(positive_filename, "w", encoding="utf-8")
    negative_file = open(negative_filename, "w", encoding="utf-8")

    modified_recommendations = []

    for recommendation in recommendations:
        if "entire_row" in recommendation:
            entire_row = recommendation["entire_row"].copy()

            # Process Cleaned_Positive_Summmary
            if "Cleaned_Positive_Summmary" in entire_row:
                positive_summary = entire_row["Cleaned_Positive_Summmary"]
                sentences = sent_tokenize(positive_summary)
                if len(sentences) > 0:
                    entire_row["Cleaned_Positive_Summmary"] = sentences[0]
                    positive_file.write(f'"{sentences[0]}", ')

            # Process Cleaned_Negative_Summmary
            if "Cleaned_Negative_Summmary" in entire_row:
                negative_summary = entire_row["Cleaned_Negative_Summmary"]
                sentences = sent_tokenize(negative_summary)
                if len(sentences) > 0:
                    entire_row["Cleaned_Negative_Summmary"] = sentences[0]
                    negative_file.write(f'"{sentences[0]}", ')

            modified_recommendations.append({"entire_row": entire_row})

    positive_file.close()
    negative_file.close()

    return modified_recommendations
