import os
import sys
import json
import time
import logging
import pandas as pd
import importlib.util
from pathlib import Path
from typing import Dict, List, Any
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_experiments.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add the project root directory to Python path to fix imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
logger.info(f"Added project root to Python path: {PROJECT_ROOT}")

# Path to your app config file - modify as needed
CONFIG_PATH = Path("/home/ib-developer/Windsurf projects/grandma_remedy/rag_func/test_config.py")
CONFIG_MODULE_NAME = "rag_func.test_config.py"

# Path to your app.py
APP_PATH = Path("rag_func/app.py")

# Google Sheets scopes
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Predefined evaluation questions and ground truths
EVALUATION_QUESTIONS = [
    {
        "question": "What herbs can help with a headache?",
        "ground_truth": "Ginger, peppermint, feverfew, and willow bark can help with headaches."
    },
    {
        "question": "How can I treat a fever naturally?",
        "ground_truth": "Cold compresses, staying hydrated, and rest can help treat a fever naturally."
    },
    {
        "question": "What is a good remedy for a sore throat?",
        "ground_truth": "Honey, warm salt water gargles, and herbal teas are good remedies for a sore throat."
    },
    # Add more evaluation questions as needed
]


def authenticate_google_sheets() -> gspread.Client:
    try:
        # First try service account authentication
        try:
            creds_file = os.environ.get("GOOGLE_SHEETS_CREDS", "credentials.json")
            logger.info(f"Looking for credentials file at: {creds_file}")

            if not os.path.exists(creds_file):
                logger.warning(f"Credentials file not found at: {creds_file}")
                # Try absolute path if relative path fails
                if not os.path.isabs(creds_file):
                    abs_creds_file = os.path.join(PROJECT_ROOT, creds_file)
                    logger.info(f"Trying absolute path: {abs_creds_file}")
                    if os.path.exists(abs_creds_file):
                        creds_file = abs_creds_file
                        logger.info(f"Found credentials at absolute path: {creds_file}")
                    else:
                        logger.warning(f"Credentials file not found at absolute path either: {abs_creds_file}")

            credentials = Credentials.from_service_account_file(
                creds_file, scopes=SCOPES
            )
            client = gspread.authorize(credentials)
            logger.info("Successfully authenticated with Google Sheets using service account")
            # Test the connection
            _ = client.list_spreadsheet_files()
            logger.info("Successfully listed spreadsheet files - connection is working")
            return client
        except FileNotFoundError as e:
            # Fall back to gspread OAuth authentication flow if service account fails
            logger.warning(f"Service account credentials not found: {e}, trying OAuth flow")
            from gspread_oauth import GoogleCredentials

            client = GoogleCredentials(
                scopes=SCOPES,
                credentials_dir=os.path.expanduser("~/.config/gspread_oauth")
            ).authorize()
            logger.info("Successfully authenticated with Google Sheets using OAuth")
            return client
        except Exception as e:
            logger.error(f"Service account authentication failed: {e}")
            raise
    except Exception as e:
        logger.error(f"Failed to authenticate with Google Sheets: {e}")
        logger.info("Authentication troubleshooting tips:")
        logger.info("1. Make sure your credentials.json file exists and is valid")
        logger.info("2. Ensure you've shared the spreadsheet with your service account email")
        logger.info("3. Verify the spreadsheet URL is correct")
        sys.exit(1)


def read_config_permutations(sheet_url: str, sheet_name: str = "Sheet1") -> pd.DataFrame:
    try:
        client = authenticate_google_sheets()

        try:
            logger.info(f"Attempting to open spreadsheet by URL: {sheet_url}")
            # Try to open the spreadsheet by URL
            spreadsheet = client.open_by_url(sheet_url)
        except Exception as e:
            logger.warning(f"Could not open by URL: {e}")
            # Extract sheet ID from the URL
            sheet_id = sheet_url.split('/d/')[1].split('/')[0] if '/d/' in sheet_url else sheet_url
            logger.info(f"Trying to open by key: {sheet_id}")
            spreadsheet = client.open_by_key(sheet_id)

        # Select the worksheet
        logger.info(f"Attempting to open worksheet: {sheet_name}")
        worksheet = spreadsheet.worksheet(sheet_name)

        # Check if we can access the worksheet
        logger.info(f"Worksheet title: {worksheet.title}")
        logger.info(f"Worksheet ID: {worksheet.id}")

        # Get all values as a list of lists
        logger.info("Fetching all values from worksheet")
        data = worksheet.get_all_values()

        if not data:
            logger.error("Spreadsheet appears to be empty")
            sys.exit(1)

        # Log first few rows to debug
        logger.info(f"First row (headers): {data[0]}")
        if len(data) > 1:
            logger.info(f"Second row (first data row): {data[1]}")

        # Convert to DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])

        # Only include rows that have values in the configuration columns
        df = df.dropna(how='all')
        df = df[df.iloc[:, 0] != ""]  # Filter out rows where the first column is empty

        logger.info(f"Read {len(df)} configuration permutations from the Google Sheet")
        logger.info(f"Columns found: {', '.join(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Failed to read configuration permutations: {e}")
        logger.info(f"Sheet URL attempted: {sheet_url}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def update_app_config(config_values: Dict[str, Any]) -> None:
    try:
        # Instead of importing the module which can cause import errors,
        # let's try to extract the APP_CONFIG directly from the file
        app_config = {}

        # Check if the config file exists
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    content = f.read()

                # Try to find and extract the APP_CONFIG definition
                import re
                config_match = re.search(r"APP_CONFIG\s*=\s*({[^}]*})", content, re.DOTALL)
                if config_match:
                    # Use a safer approach to evaluate the dictionary
                    config_str = config_match.group(1).strip()
                    try:
                        # Try to safely evaluate the Python dictionary
                        import ast
                        app_config = ast.literal_eval(config_str)
                        logger.info("Successfully extracted APP_CONFIG from file")
                    except (SyntaxError, ValueError) as e:
                        logger.warning(f"Could not parse APP_CONFIG: {e}")
                        app_config = {}
                else:
                    logger.warning("APP_CONFIG not found in the config file")
            except Exception as e:
                logger.warning(f"Error reading config file: {e}")
        else:
            logger.warning(f"Config file not found at {CONFIG_PATH}, will create a new one")

        column_to_config = {
            "Embeddings": "embedding_model",
            "vector_store": "vector_store_type",
            "retrieval": "retriever_type",
            "llm": "llm_model",
            "reranking": "reranker_type",
            "chunking": "chunking_strategy",
        }

        # Create a new dictionary with the mapped keys
        mapped_config = {}
        for sheet_key, value in config_values.items():
            if sheet_key in column_to_config and value:
                config_key = column_to_config[sheet_key]
                mapped_config[config_key] = value
            else:
                logger.debug(f"Skipping unmapped sheet key: {sheet_key}")

        logger.info(f"Mapped configuration: {mapped_config}")

        # Update only the keys that are in the mapped dictionary
        for key, value in mapped_config.items():
            if key in app_config:
                # Handle type conversion for known types
                if isinstance(app_config[key], bool) and isinstance(value, str):
                    app_config[key] = value.lower() == "true"
                elif isinstance(app_config[key], int) and isinstance(value, str):
                    try:
                        app_config[key] = int(value)
                    except ValueError:
                        logger.warning(f"Could not convert {value} to int for {key}")
                elif isinstance(app_config[key], float) and isinstance(value, str):
                    try:
                        app_config[key] = float(value)
                    except ValueError:
                        logger.warning(f"Could not convert {value} to float for {key}")
                else:
                    app_config[key] = value
            else:
                # If the key doesn't exist in app_config, add it anyway
                app_config[key] = value
                logger.warning(f"Config key {key} not found in APP_CONFIG, adding it")

        # Construct updated content with the new APP_CONFIG
        updated_content = f"""# Auto-generated configuration file
# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}

APP_CONFIG = {json.dumps(app_config, indent=4, ensure_ascii=False)}
"""

        # Write the updated content to the config file
        with open(CONFIG_PATH, "w") as f:
            f.write(updated_content)

        logger.info(f"Updated app_config at {CONFIG_PATH} with new configuration values")
    except Exception as e:
        logger.error(f"Failed to update app_config: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def run_evaluation_with_questions(questions: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Run the RAG evaluation with the given questions.
    """
    try:
        # Path to the app module
        app_path = os.path.join(PROJECT_ROOT, APP_PATH)
        logger.info(f"Looking for app module at: {app_path}")

        if not os.path.exists(app_path):
            logger.error(f"App file not found at {app_path}")
            # Try to find app.py elsewhere
            possible_paths = list(PROJECT_ROOT.glob("**/app.py"))
            if possible_paths:
                logger.info(f"Found potential app.py at: {possible_paths[0]}")
                app_path = str(possible_paths[0])
            else:
                return {"error": "App file not found"}

        try:
            # Import the app module
            logger.info(f"Importing app module from: {app_path}")
            spec = importlib.util.spec_from_file_location("app", app_path)
            app_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(app_module)

            # Verify required functions exist
            required_functions = ["initialize_rag_system", "process_query_with_rag", "evaluate_response"]
            for func in required_functions:
                if not hasattr(app_module, func):
                    logger.error(f"Missing required function in app module: {func}")
                    return {"error": f"Missing function in app module: {func}"}

            # Initialize the RAG system
            logger.info("Initializing RAG system...")
            rag_system = app_module.initialize_rag_system()
            logger.info("RAG system initialized successfully")

            # List to store results for each question
            all_metrics = []

            # Metrics mapping for consistent column names
            metric_mapping = {
                "faithfulness": "Faithfulness",
                "answer_relevance": "Answer Relevancy",
                "groundedness": "Groundedness",
                "context_relevance": "Context relevance"
            }

            # Process each question
            for question_data in questions:
                question = question_data["question"]
                ground_truth = question_data["ground_truth"]

                logger.info(f"Evaluating question: {question}")

                # Use the RAG pipeline to process the query
                logger.info("Calling process_query_with_rag")
                response, reranked_docs = app_module.process_query_with_rag(rag_system, question, chat_history="")
                logger.info(f"Response received. Length: {len(response)}")
                logger.info(f"Num documents: {len(reranked_docs) if reranked_docs else 0}")

                # Evaluate the response
                if reranked_docs:
                    logger.info("Calling evaluate_response")
                    metrics = app_module.evaluate_response(
                        rag_system, reranked_docs, question, response
                    )

                    # MODIFIED: Handle EvaluationResult object - extract metrics based on object attributes
                    mapped_metrics = {}

                    # Option 1: If metrics has a to_dict() or dict() method
                    try:
                        if hasattr(metrics, 'to_dict'):
                            metrics_dict = metrics.to_dict()
                        elif hasattr(metrics, 'dict'):
                            metrics_dict = metrics.dict()
                        else:
                            # Option 2: Access attributes directly
                            metrics_dict = {}
                            # Try to access common metric attributes
                            for internal_key in metric_mapping.keys():
                                if hasattr(metrics, internal_key):
                                    metrics_dict[internal_key] = getattr(metrics, internal_key)

                            # If still empty, try getting all attributes
                            if not metrics_dict:
                                # Get all non-private attributes (those not starting with _)
                                metrics_dict = {attr: getattr(metrics, attr)
                                                for attr in dir(metrics)
                                                if not attr.startswith('_') and not callable(getattr(metrics, attr))}
                    except Exception as e:
                        logger.warning(f"Failed to extract metrics dictionary: {e}")
                        # Fallback: If all else fails, create a basic dictionary with zero values
                        metrics_dict = {key: 0.0 for key in metric_mapping.keys()}

                    # Map the metrics to match spreadsheet columns
                    for internal_key, sheet_key in metric_mapping.items():
                        mapped_metrics[sheet_key] = metrics_dict.get(internal_key, 0.0)

                    all_metrics.append(mapped_metrics)
                    logger.info(f"Metrics for question: {mapped_metrics}")
                else:
                    logger.warning(f"No documents retrieved for question: {question}")
                    # Add zero metrics since no docs were retrieved
                    mapped_metrics = {sheet_key: 0.0 for internal_key, sheet_key in metric_mapping.items()}
                    all_metrics.append(mapped_metrics)

            # Aggregate metrics across all questions
            if not all_metrics:
                logger.error("No metrics collected during evaluation")
                return {"error": "No metrics collected"}

            aggregated_metrics = {}
            for key in metric_mapping.values():  # Use the mapped keys
                values = [m.get(key, 0) for m in all_metrics if key in m]
                if values:
                    aggregated_metrics[key] = sum(values) / len(values)
                else:
                    aggregated_metrics[key] = 0

            logger.info(f"Aggregated metrics: {aggregated_metrics}")
            return aggregated_metrics

        except Exception as e:
            logger.error(f"Error executing RAG evaluation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": f"Evaluation error: {str(e)}"}

    except Exception as e:
        logger.error(f"Failed to run evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def write_results_to_sheet(sheet_url: str, sheet_name: str, row_index: int,
                           metrics: Dict[str, float]) -> None:
    try:
        client = authenticate_google_sheets()

        # Open the spreadsheet
        logger.info(f"Opening spreadsheet to write results: {sheet_url}")
        try:
            spreadsheet = client.open_by_url(sheet_url)
        except Exception as e:
            logger.warning(f"Could not open by URL: {e}")
            # Extract sheet ID from the URL
            sheet_id = sheet_url.split('/d/')[1].split('/')[0] if '/d/' in sheet_url else sheet_url
            logger.info(f"Trying to open by key: {sheet_id}")
            spreadsheet = client.open_by_key(sheet_id)

        worksheet = spreadsheet.worksheet(sheet_name)

        # Get all headers
        headers = worksheet.row_values(1)
        logger.info(f"Sheet headers: {headers}")
        logger.info(f"Metrics to write: {metrics}")

        # Prepare updates
        updates = []
        for key, value in metrics.items():
            if key in headers:
                column_index = headers.index(key) + 1  # 1-based indexing
                # Format the value to 4 decimal places if it's a float
                if isinstance(value, float):
                    value = round(value, 4)
                cell_addr = gspread.utils.rowcol_to_a1(row_index + 2, column_index)
                logger.info(f"Updating cell {cell_addr} with value {value} for metric {key}")
                updates.append({
                    "range": cell_addr,
                    "values": [[value]]
                })
            else:
                logger.warning(f"Metric {key} not found in headers, skipping")

        # Batch update the cells
        if updates:
            logger.info(f"Sending batch update with {len(updates)} updates")
            worksheet.batch_update(updates)
            logger.info(f"Updated row {row_index + 2} with evaluation metrics")
        else:
            logger.warning("No metrics to update in the sheet")

        # Verify the updates
        time.sleep(1)  # Give the API time to process
        updated_row = worksheet.row_values(row_index + 2)
        logger.info(f"After update, row contains: {updated_row}")

    except Exception as e:
        logger.error(f"Failed to write results to Google Sheet: {e}")
        import traceback
        logger.error(traceback.format_exc())


def main() -> None:
    sheet_url = os.environ.get("GOOGLE_SHEET_URL")
    if not sheet_url:
        logger.error("Google Sheet URL not found in environment variables")
        print("Please provide a Google Sheet URL via GOOGLE_SHEET_URL environment variable")
        print(
            "Example usage: GOOGLE_SHEET_URL='https://docs.google.com/spreadsheets/d/1YNJx2flO7TfeZRhs-sA85kTznmCsKPRuQTdVkfPTwXw/edit' python rag_experiment_runner.py")
        sys.exit(1)

    sheet_name = os.environ.get("GOOGLE_SHEET_TAB", "Sheet1")

    logger.info(f"Connecting to Google Sheet: {sheet_url}")
    logger.info(f"Using sheet tab: {sheet_name}")

    # Read configuration permutations
    df = read_config_permutations(sheet_url, sheet_name)

    if df.empty:
        logger.error("No configuration permutations found in the sheet")
        sys.exit(1)

    # Process each permutation
    for index, row in df.iterrows():
        logger.info(f"Processing permutation {index + 1}/{len(df)}")
        logger.info(f"Configuration: {row.to_dict()}")

        result_columns = ["Faithfulness", "Answer Relevancy", "Groundedness", "Context relevance", "status",
                          "timestamp", "error"]
        config_values = {k: v for k, v in row.items() if pd.notna(v) and k not in result_columns}

        try:
            # Update status to "Running"
            write_results_to_sheet(
                sheet_url,
                sheet_name,
                index,
                {"status": "Running", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            )

            # Update app_config with the current permutation
            update_app_config(config_values)

            # Run the evaluation
            metrics = run_evaluation_with_questions(EVALUATION_QUESTIONS)

            # Check for errors
            if "error" in metrics:
                logger.error(f"Evaluation returned an error: {metrics['error']}")
                write_results_to_sheet(
                    sheet_url,
                    sheet_name,
                    index,
                    {"status": "Failed", "error": metrics["error"], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
                )
                continue

            # Add timestamp to metrics
            metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            metrics["status"] = "Completed"

            # Write results back to the sheet
            write_results_to_sheet(sheet_url, sheet_name, index, metrics)

            logger.info(f"Completed permutation {index + 1}")
        except Exception as e:
            logger.error(f"Error processing permutation {index + 1}: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Update status to "Failed" in the sheet
            write_results_to_sheet(
                sheet_url,
                sheet_name,
                index,
                {"status": "Failed", "error": str(e)[:100], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            )

        # Pause briefly to avoid rate limits
        time.sleep(2)

    logger.info("All permutations processed successfully")


if __name__ == "__main__":
    main()