import os
import gradio as gr
import requests
import inspect
import pandas as pd
import re
import mimetypes

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
questions_url = f"{DEFAULT_API_URL}/questions"
submit_url = f"{DEFAULT_API_URL}/submit"
random_question_url = f"{DEFAULT_API_URL}/random-question"
get_file_url = f"{DEFAULT_API_URL}/files/"  # append task_id to this


# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")

    def __call__(self, question: str, path: str | None) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}")
        if path:
            print(f"Question requires file present at path: {path}")

        fixed_answer = "This is a default answer."
        print(f"Agent returning fixed answer: {fixed_answer}")
        return fixed_answer


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


def get_task_file(task_id: str, save_dir="downloads"):
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

        print("\n" + "-" * 30 + f"|START {i}|" + "-" * 30)
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue

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

        print(f"Running agent on question: {item}")

        try:
            submitted_answer = agent(question_text, fetched_path)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            item["submitted_answer"] = submitted_answer
        except Exception as e:
            err_msg = f"AGENT ERROR on task {task_id}: {e}"
            print(err_msg)
            item["submitted_answer"] = err_msg

        results_log.append(item)
        print("-" * 30 + f"|END {i}|" + "-" * 30 + "\n")

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

    # evaluation buttons
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
