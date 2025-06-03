import json
import pandas as pd
from flatten_json import flatten
import argparse
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials


def process_json_to_gsheet(input_file, spreadsheet_name, spreadsheet_id=None):
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Extract the components
    app_config = data.get('app_config', {})
    metrics = data.get('metrics', {})
    questions = data.get('questions', [])

    # Flatten app_config and metrics dictionaries
    flat_app_config = flatten(app_config, separator='_')
    flat_metrics = flatten(metrics, separator='_')

    # Create a list to hold all rows
    rows = []

    # Process each question
    for question_obj in questions:
        row = {}
        scores = question_obj.get('scores', {})

        row['embedding'] = app_config.get('embedding', '')
        row['vector_store'] = app_config.get('vector_store', '')
        row['retrieval'] = app_config.get('retrieval', '')
        row['llm'] = app_config.get('llm', '')

        row['evaluation'] = app_config.get('evaluation', '')
        row['reranking'] = app_config.get('reranking', '')
        row['chunking'] = app_config.get('chunking', '')

        row['faithfulness'] = metrics.get('faithfulness', '')
        row['answer_relevancy'] = metrics.get('answer_relevancy', '')
        row['context_relevance'] = metrics.get('context_relevance', '')
        row['response_groundedness'] = metrics.get('response_groundedness', '')

        row['score_faithfulness'] = scores.get('faithfulness', '')
        row['score_answer_relevancy'] = scores.get('answer_relevancy', '')
        row['score_response_groundedness'] = scores.get('response_groundedness', '')
        row['score_context_relevance'] = scores.get('context_relevance', '')

        rows.append(row)

    questions_df = pd.DataFrame(rows)

    column_order = [
        'embedding', 'vector_store', 'retrieval', 'llm', 'evaluation', 'reranking',	'chunking',	'faithfulness',
        'answer_relevancy',	'response_groundedness', 'context_relevance', 'score_faithfulness', 'score_answer_relevancy',
        'score_response_groundedness', 'score_context_relevance'
    ]

    for col in column_order:
        if col not in questions_df.columns:
            questions_df[col] = ''

    questions_df = questions_df[column_order]

    summary_data = {
        'Parameter': list(flat_app_config.keys()) + list(flat_metrics.keys()),
        'Value': list(flat_app_config.values()) + list(flat_metrics.values()),
        'Type': ['Config'] * len(flat_app_config) + ['Metric'] * len(flat_metrics)
    }
    summary_df = pd.DataFrame(summary_data)

    # Set up Google Sheets authentication
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

    credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
    client = gspread.authorize(credentials)

    # Open or create spreadsheet
    if spreadsheet_id:
        try:
            # Open by ID if provided
            spreadsheet = client.open_by_key(spreadsheet_id)
            print(f"Opened existing spreadsheet by ID: {spreadsheet_id}")
        except gspread.SpreadsheetNotFound:
            print(f"Error: Spreadsheet with ID {spreadsheet_id} not found")
            return
    else:
        try:
            # Fall back to opening by name
            spreadsheet = client.open(spreadsheet_name)
            print(f"Opened existing spreadsheet: {spreadsheet_name}")
        except gspread.SpreadsheetNotFound:
            spreadsheet = client.create(spreadsheet_name)
            print(f"Created new spreadsheet: {spreadsheet_name}")

    # Debug print to examine the data before uploading
    print(f"First row of data to be uploaded: {questions_df.iloc[0].to_dict()}")

    # Get the Questions sheet
    try:
        questions_sheet = spreadsheet.worksheet("Questions")

        # Get existing data to determine where to append
        existing_data = questions_sheet.get_all_values()
        start_row = len(existing_data) + 1

        # Prepare data for batch update
        new_data = questions_df.values.tolist()
        if start_row == 1:  # If sheet is empty, include headers
            headers = questions_df.columns.tolist()
            new_data.insert(0, headers)

        # Update sheet with new data
        cell_range = f'A{start_row}'
        questions_sheet.update(cell_range, new_data)

    except gspread.WorksheetNotFound:
        # If sheet doesn't exist, create it
        questions_sheet = spreadsheet.add_worksheet(title="Questions", rows=len(questions_df) + 1,
                                                    cols=len(questions_df.columns))

        # Add headers and data
        headers = questions_df.columns.tolist()
        questions_sheet.update('A1', [headers])
        questions_sheet.update('A2', questions_df.values.tolist())

    # Update or create Summary sheet
    try:
        summary_sheet = spreadsheet.worksheet("Summary")
        existing_summary = summary_sheet.get_all_values()

        # If summary exists with data, append new data
        if len(existing_summary) > 1:
            start_row = len(existing_summary) + 1
            summary_sheet.update(f'A{start_row}', summary_df.values.tolist())
        else:
            # Replace empty summary sheet
            headers = summary_df.columns.tolist()
            summary_sheet.update('A1', [headers])
            summary_sheet.update('A2', summary_df.values.tolist())

    except gspread.WorksheetNotFound:
        summary_sheet = spreadsheet.add_worksheet(title="Summary", rows=len(summary_df) + 1,
                                                  cols=len(summary_df.columns))
        headers = summary_df.columns.tolist()
        summary_sheet.update('A1', [headers])
        summary_sheet.update('A2', summary_df.values.tolist())

    print(f"Data successfully uploaded to Google Sheets: {spreadsheet.title}")
    print(f"Spreadsheet URL: https://docs.google.com/spreadsheets/d/{spreadsheet.id}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSON evaluation data to Google Sheets")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("-s", "--spreadsheet", help="Name of the Google Spreadsheet",
                        default="Evaluation Results")
    parser.add_argument("-id", "--spreadsheet_id", help="ID of an existing Google Spreadsheet (from URL)",
                        default=None)
    parser.add_argument("--append", help="Append data instead of replacing",
                        action="store_true")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return

    process_json_to_gsheet(args.input_file, args.spreadsheet, args.spreadsheet_id)


if __name__ == "__main__":
    main()