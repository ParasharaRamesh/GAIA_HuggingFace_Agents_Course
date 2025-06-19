import os
import gradio as gr
import requests
import pandas as pd
import re
import mimetypes
import json
import uuid
from langchain_core.messages import HumanMessage, AIMessage

from agents.workflow import create_worfklow
from tools.audio import model # this triggers the whisper model to be loaded
from agents.state import AgentState
from langfuse.langchain import CallbackHandler #


# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
questions_url = f"{DEFAULT_API_URL}/questions"
submit_url = f"{DEFAULT_API_URL}/submit"
random_question_url = f"{DEFAULT_API_URL}/random-question"
get_file_url = f"{DEFAULT_API_URL}/files/"  # append task_id to this

with open('expected_answers.json', 'r', encoding='utf-8') as f:
    expected_answers = {item["task_id"]: item["Final answer"] for item in json.load(f)}


# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
class BasicAgent:
    def __init__(self):
        print("BasicAgent initializing with LLMs and workflow...")

        self.langfuse_handler = CallbackHandler()
        print("Langfuse callback handler initialized.")

        # --- Create the Master Orchestrator Workflow ---
        print("Creating master orchestrator workflow...")
        orchestrator_compiled_app = create_worfklow()
        self.orchestrator_app = orchestrator_compiled_app.with_config(
            {"callbacks": [self.langfuse_handler]}
        )
        print("Langfuse callback handler created.")

    def __call__(self, question: str, path: str | None) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}")
        print(f"Path: {path}")

        # Combine question and path into the initial HumanMessage for the orchestrator
        # The orchestrator will then delegate based on this combined input.
        full_input_content = question
        if path:
            # If a path is provided, embed it in the message for relevant agents
            # e.g., "Analyze the image located at path/to/image.png: [original question]"
            full_input_content = f"The query relates to a file at '{path}'. Please incorporate this context. Original query: {question}"

        initial_state: AgentState = { # Explicitly type as AgentState
            "query": full_input_content, # The overall query
            "messages": [HumanMessage(content=full_input_content)], # Initial message to the orchestrator
            "final_answer": None # Initialize final_answer
        }

        print(f"\n--- Running orchestrator workflow for: '{full_input_content}' ---")

        try:
            # Invoke the workflow. We are using .invoke() here for simplicity
            # For streaming or more detailed progress, you might iterate over .stream()
            final_state: AgentState = self.orchestrator_app.invoke(initial_state)

            # Extract the final answer from the state
            if final_state.get("final_answer"):
                print(f"\n--- Orchestrator Final Answer ---")
                return final_state["final_answer"]
            else:
                print("\n--- Orchestrator completed, but no explicit 'final_answer' was extracted. ---")
                # Fallback: Return the content of the last AI message if no specific final_answer was set
                if final_state.get("messages"):
                    last_message = final_state["messages"][-1]
                    if isinstance(last_message, AIMessage):
                        return last_message.content
                return "Workflow completed, but no clear answer was found."

        except Exception as e:
            print(f"Error during workflow execution: {e}")
            return f"An error occurred while processing your request: {e}"
        finally:
            print("Flushing Langfuse traces...")
            self.langfuse_handler.langfuse.flush()
            print("Langfuse traces flushed.")


# Helper methods

def get_agent_code_link():
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    space_id = os.getenv("SPACE_ID")  # Get the SPACE_ID for sending link to the code
    try:
        if space_id:
            agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
            print(f"Agent code is present @ {agent_code}")
            return agent_code
        return None
    except Exception as e:
        print(f"Error getting agent code link: {e}")
        return None


def fetch_questions():
    print(f"\nFetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            err_msg = "Fetched questions list is empty or invalid format."
            print(err_msg)
            return None, err_msg

        print(f"Fetched {len(questions_data)} questions.\n")
        return questions_data, None
    except requests.exceptions.RequestException as e:
        err_msg = f"Error fetching questions: {e}"
        print(err_msg)
        return None, err_msg
    except requests.exceptions.JSONDecodeError as e:
        err_msg = f"Error decoding JSON response from questions endpoint: {e} | Response text: {response.text[:500]}"
        print(err_msg)
        return None, err_msg
    except Exception as e:
        err_msg = f"Unexpected error fetching questions: {e}"
        print(err_msg)
        return None, err_msg


def get_random_question():
    print(f"\nFetching a random question from: {random_question_url}")
    try:
        response = requests.get(random_question_url, timeout=15)
        response.raise_for_status()
        question_data = response.json()
        if not question_data:
            err_msg = "Fetched random question is empty or in invalid format."
            print(err_msg)
            return None, err_msg
        print(f"Fetched random question with task_id: {question_data.get('task_id', 'N/A')}\n")
        return question_data, None
    except requests.exceptions.RequestException as e:
        err_msg = f"Error fetching random question: {e}\n"
        print(err_msg)
        return None, err_msg
    except requests.exceptions.JSONDecodeError as e:
        err_msg = f"Error decoding JSON response from random question endpoint: {e} | Response text: {response.text[:500]}\n"
        print(err_msg)
        return None, err_msg
    except Exception as e:
        err_msg = f"Unexpected error fetching random question: {e}\n"
        print(err_msg)
        return None, err_msg


def evaluate_random_question(profile: gr.OAuthProfile | None):
    """
    Fetches a random question, runs the BasicAgent on it,
    and displays the result.
    """
    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    agent_code = get_agent_code_link()
    if not agent_code:
        return "Error getting the agent code link from the SPACE_ID environment variable", None

    # Instantiate Agent
    try:
        agent = BasicAgent()
    except Exception as e:
        error_msg = f"Error instantiating agent: {e}"
        return error_msg, None

    # Fetch random question
    random_question, error_msg = get_random_question()
    if error_msg:
        return error_msg, None

    # Wrap random_question in a list
    answers_payload, results_log = run_agent(agent, [random_question])

    if not answers_payload or len(answers_payload) == 0:
        print("Agent did not produce any answer for the random question.\n")
        return "Agent did not produce any answer for the random question.", pd.DataFrame(results_log)

    # No submission to server for a single random question.
    status_message = "✅ Agent evaluated the random question successfully!"
    results_df = pd.DataFrame(results_log)
    return status_message, results_df


def evaluate_custom_question(profile: gr.OAuthProfile | None, custom_question_text: str, question_id_input: str):
    """
    Runs the BasicAgent on a custom question provided by the user.
    """
    local_file_path = None

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    agent_code = get_agent_code_link()
    if not agent_code:
        return "Error getting the agent code link from the SPACE_ID environment variable", None

    if not custom_question_text or len(custom_question_text.strip()) < 5:
        return "Please enter a custom question of at least 5 characters.", None

    status_message_for_return = ''
    local_file_path = None

    if question_id_input and question_id_input.strip():
        mock_task_id = question_id_input.strip()
        print(f"Attempting to get file for Question ID: {mock_task_id}...")

        # Call the provided get_task_file function
        local_file_path = get_task_file(mock_task_id)

        if local_file_path:
            print(f"File ready at: {local_file_path}")
            status_message_for_return += f"File for ID {mock_task_id} processed. "
        else:
            print(f"Could not get file for ID {mock_task_id}. Proceeding without file context.")
            status_message_for_return += f"NOTE: Failed to get file for ID {mock_task_id}. "
    else:
        mock_task_id = f"custom_{uuid.uuid4()}"  # Generate unique ID if no question_id
        print(f"Generated new mock_task_id: {mock_task_id}")

    # Create a mock question data for the custom question
    mock_question_data = [{
        "task_id": mock_task_id,
        "question": custom_question_text,
        "file_name": local_file_path   # Custom questions typically don't have associated files unless manually handled
    }]

    print(f"\nRunning agent on custom question (task_id: {mock_task_id}): {custom_question_text[:100]}...\n")
    # Instantiate Agent
    try:
        agent = BasicAgent()  # This will be your full agent system later
    except Exception as e:
        error_msg = f"Error instantiating agent: {e}"
        return error_msg, None

    # Run the agent with the mock question
    answers_payload, results_log = run_agent(agent, mock_question_data)

    if not answers_payload or len(answers_payload) == 0:
        print("Agent did not produce any answer for the custom question.\n")
        return "Agent did not produce any answer for the custom question.", pd.DataFrame(results_log)

    # No submission to server for a custom question.
    status_message = "✅ Agent evaluated the custom question successfully!"
    results_df = pd.DataFrame(results_log)
    return status_message, results_df


def get_task_file(task_id: str, save_dir="."):
    """
    Downloads the file associated with the given task_id and saves it locally.
    If no filename is provided by the server, uses the content type to determine an extension.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1️⃣ Check if any file with the task_id prefix already exists
    for filename in os.listdir(save_dir):
        if filename.startswith(task_id):
            existing_path = os.path.join(save_dir, filename)
            print(f"File already exists for task {task_id}: {existing_path}")
            return existing_path  # Skip downloading, reuse existing file

    # 2️⃣ Download if no existing file
    file_url = f"{get_file_url}{task_id}"
    try:
        print(f"Attempting to download file for task {task_id} from {file_url}...")
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()

        # 1. Extract filename from Content-Disposition using regex for robustness
        content_disp = response.headers.get("Content-Disposition", "")
        filename = None
        match = re.search(r'filename="?([^";]+)"?', content_disp, re.IGNORECASE)

        if match:
            filename = match.group(1)
        else:
            # 2. Fallback: use content type for extension
            content_type = response.headers.get("Content-Type", "").split(";")[0]
            extension = mimetypes.guess_extension(content_type) or ""
            filename = f"{task_id}_file{extension}"

        local_path = os.path.join(save_dir, filename)
        with open(local_path, "wb") as f:
            f.write(response.content)

        print(f"File downloaded and saved to: {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading file for task {task_id}: {e}")
        return None


def run_agent(agent, questions_data):
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...\n")

    for i, item in enumerate(questions_data):
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name")

        print("\n" + "-" * 30 + f"|START {i+1}|" + "-" * 30)
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue

        expected_answer = expected_answers.get(task_id, "")
        item["expected_answer"] = expected_answer

        # ✅ Check if there is an associated file and attempt to fetch it
        fetched_path = None
        if file_name and len(file_name.strip()) > 1:
            fetched_path = get_task_file(task_id)  # Function we defined earlier
            if not fetched_path:
                err_msg = f"FILE DOWNLOAD ERROR for task {task_id}"
                print(err_msg)
                item["submitted_answer"] = err_msg
                results_log.append(item)
                continue  # Skip this task and move to next

        print(f"Running agent on question: {question_text}. \n")
        print(f"Question has file associated with it ? : {fetched_path != None} \n")

        try:
            submitted_answer = agent(question_text, fetched_path)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            item["submitted_answer"] = submitted_answer
        except Exception as e:
            err_msg = f"AGENT ERROR on task {task_id}: {e}"
            print(err_msg)
            item["submitted_answer"] = err_msg

        results_log.append(item)
        print("-" * 30 + f"|END {i+1}|" + "-" * 30 + "\n")

    print(f"Finished running agent on {len(questions_data)} questions...!\n")
    return answers_payload, results_log


def submit_answers(username, agent_code, answers_payload):
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload
    }
    print(f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}' to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful :) !!")
        return final_status
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        return status_message
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        return status_message
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        return status_message
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        return status_message


# Main method
def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # Find out the logged in person
    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}\n")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    # 0.  Determine HF Space Runtime URL and Repo URL
    agent_code = get_agent_code_link()
    if not agent_code:
        return "Error getting the agent code link from the SPACE_ID environment variable", None

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = BasicAgent()
    except Exception as e:
        error_msg = f"Error instantiating agent: {e}"
        return error_msg, None

    # 2. Fetch Questions
    questions_data, error_msg = fetch_questions()
    if error_msg:
        return error_msg, None

    # 3. Run your Agent
    answers_payload, results_log = run_agent(agent, questions_data)

    if not answers_payload or len(answers_payload) == 0:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Submit answers
    final_status = submit_answers(username, agent_code, answers_payload)
    results_df = pd.DataFrame(results_log)
    return final_status, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    gr.Markdown("---")  # Optional separator for clarity
    gr.Markdown("### Test Your Agent with a Custom Question")
    custom_question_input = gr.Textbox(
        label="Enter your custom question here:",
        placeholder="e.g., 'What are the main features of the latest iPhone model?'",
        lines=3
    )
    question_id_input = gr.Textbox(
        label="Optional: Enter Question ID (for fetching associated files)",
        placeholder="e.g., 1 or 20 (This will download a file if available and pass its path to the agent)",
        lines=1
    )
    run_custom_button = gr.Button("Run Agent on Custom Question")

    run_button = gr.Button("Run Evaluation & Submit All Answers")
    run_random_button = gr.Button("Evaluate on Random Question")

    # button results
    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    # button callbacks
    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

    run_random_button.click(
        fn=evaluate_random_question,
        outputs=[status_output, results_table]
    )

    run_custom_button.click(
        fn=evaluate_custom_question,
        inputs=[custom_question_input, question_id_input],
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")  # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:  # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-" * (60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)
