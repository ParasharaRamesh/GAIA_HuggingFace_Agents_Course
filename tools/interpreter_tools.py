import os
import subprocess
import uuid
from typing import Dict, Any, Optional, List
from langchain_core.tools import tool


# --- File System Tools ---

@tool
def read_file(file_path: str) -> Dict[str, str]:
    """
    Reads the full content of a text file from the local filesystem.

    Args:
        file_path (str): The relative path to the file to be read.
                         Example: 'my_script.py' or 'data/input.csv'.

    Returns:
        Dict[str, str]: A dictionary containing the result.
                        On success: {'file_content': '...the full content of the file...'}.
                        On failure: {'error': 'Error message...'}.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"file_content": f"Content of '{file_path}':\n```\n{content}\n```"}
    except FileNotFoundError:
        return {"error": f"File '{file_path}' not found."}
    except Exception as e:
        return {"error": f"Error reading file '{file_path}': {str(e)}"}


# --- Code Execution Tools ---

@tool
def run_shell_command(command: str) -> Dict[str, str]:
    """
    Executes a single shell command and captures its standard output and standard error.

    Args:
        command (str): The shell command to execute. Example: 'pip install pandas'.

    Returns:
        Dict[str, str]: A dictionary containing the command's output.
                        On success: {'stdout': '...', 'stderr': '...'}.
                        On failure: {'status': 'error', 'stdout': '...', 'stderr': '...', 'exit_code': '...'}.
    """
    try:
        # Using shell=True for convenience, but be mindful of security if exposed directly to untrusted input.
        print(f"running shell command {command}")
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True,  # Raise an exception for non-zero exit codes
            timeout=120  # Timeout after 120 seconds
        )
        output = {"stdout": result.stdout, "stderr": result.stderr}
        print(f"shell output: {output}")
        return output
    except subprocess.CalledProcessError as e:
        output = {"status": "error", "stdout": e.stdout, "stderr": e.stderr, "exit_code": str(e.returncode)}
        print(f"error shell output: {output}")
        return output
    except subprocess.TimeoutExpired as e:
        output = {"status": "error", "stdout": e.stdout, "stderr": f"Command timed out after {e.timeout} seconds."}
        print(f"error shell output: {output}")
        return output
    except Exception as e:
        output = {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}
        print(f"error shell output: {output}")
        return output


@tool
def run_python_script(script_path: str) -> Dict[str, Any]:
    """
     Executes a Python script from a local file path and captures its output and any files it creates.

    Args:
        script_path (str): The relative path to the .py script to execute.

    Returns:
        Dict[str, Any]: A detailed dictionary of the execution results, containing the following keys:
                        - 'script_path' (str): The path of the script that was executed.
                        - 'status' (str): 'success' or 'error'.
                        - 'stdout' (str): The standard output from the script.
                        - 'stderr' (str): The standard error from the script. Will contain error messages if status is 'error'.
                        - 'created_files' (List[str]): A list of full paths to any new files created by the script in the current directory.
    """
    status = "error"
    stdout = ""
    stderr = ""
    created_files: List[str] = []
    current_dir = os.getcwd()

    try:
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Python script '{script_path}' not found in current directory.")

        content = read_file(script_path)
        print(f"Content read from {script_path} is {content}")

        # Get initial files in current directory before execution
        initial_files = set(os.listdir(current_dir)) if os.path.exists(current_dir) else set()

        # Execute the Python script
        result = subprocess.run(
            ["python", script_path],  # No need for basename or changing dir
            capture_output=True,
            text=True,
            timeout=180  # Timeout after 180 seconds (3 minutes)
        )

        stdout = result.stdout
        stderr = result.stderr

        if result.returncode == 0:
            status = "success"
        else:
            status = "error"

        # Detect newly created files in the current directory
        current_files = set(os.listdir(current_dir)) if os.path.exists(current_dir) else set()
        new_files = [
            os.path.join(current_dir, f) for f in current_files - initial_files
            if os.path.isfile(os.path.join(current_dir, f))
        ]
        created_files = new_files

    except FileNotFoundError as e:
        stderr = str(e)
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout
        stderr = f"Error: Script timed out after {e.timeout} seconds.\n{e.stderr}"
    except Exception as e:
        stderr = f"An unexpected error occurred: {str(e)}"

    result = {
        "script_path": script_path,
        "status": status,
        "stdout": stdout,
        "stderr": stderr,
        "created_files": created_files
    }
    print(f"run-python-script result is {result}")
    return result


@tool
def run_generated_python_code(code: str) -> Dict[str, Any]:
    """
    Executes a string of Python code by first writing it to a temporary script file
    and then running that file. This is the primary tool for solving tasks that require
    generating and running new code. The code string should be a complete, runnable script
    that prints its final result to stdout.

    Args:
        code (str): A string containing the complete Python code to be executed.

    Returns:
        Dict[str, Any]: A dictionary containing the execution result with keys for
                        'status', 'stdout', and 'stderr'.
    """
    script_filename = f"temp_script_{uuid.uuid4()}.py"
    try:
        with open(script_filename, "w", encoding="utf-8") as f:
            f.write(code)

        # We can reuse the logic from run_python_script
        return run_python_script.invoke({"script_path": script_filename})

    except Exception as e:
        return {"status": "error",
                "message": f"An unexpected error occurred while trying to write and run the code: {str(e)}"}
    finally:
        # Ensure the temporary script is always cleaned up
        if os.path.exists(script_filename):
            os.remove(script_filename)

if __name__ == '__main__':
    pass
