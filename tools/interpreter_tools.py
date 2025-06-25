import os
import subprocess
from typing import Dict, Any, Optional, List
from langchain_core.tools import tool

# --- File System Tools ---

@tool
def read_file(file_path: str) -> str:
    """
    Reads the content of a file from the local filesystem.
    File paths are relative to the current execution directory ('.').

    Args:
        file_path (str): The path to the file to read.

    Returns:
        str: The content of the file, or an error message if the file cannot be read.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return f"File '{file_path}' content:\n```\n{content}\n```"
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"

@tool
def write_file(file_path: str, content: str) -> str:
    """
    Writes content to a file on the local filesystem. If the file already exists, it will be overwritten.
    Directories will be created if they do not exist.
    File paths are relative to the current execution directory ('.').

    Args:
        file_path (str): The path to the file to write.
        content (str): The string content to write to the file.

    Returns:
        str: A confirmation message or an error message.
    """
    try:
        # Ensure the directory exists. This handles subdirectories if file_path includes them.
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Content {content} successfully written to '{file_path}'."
    except Exception as e:
        return f"Error writing to file '{file_path}': {str(e)}"

# --- Code Execution Tools ---

@tool
def run_shell_command(command: str) -> str:
    """
    Executes a shell command in the current execution environment ('.') and returns its stdout and stderr.
    Useful for installing dependencies (e.g., 'pip install package'), listing files,
    or other system interactions.

    Args:
        command (str): The shell command to execute.

    Returns:
        str: A string containing the command's stdout and stderr.
             Returns an error message if the command fails or times out.
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
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        print(f"shell output: {output}")
        return output
    except subprocess.CalledProcessError as e:
        return (f"Error: Command '{e.cmd}' failed with exit code {e.returncode}.\n"
                f"STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
    except subprocess.TimeoutExpired as e:
        return (f"Error: Command '{e.cmd}' timed out after {e.timeout} seconds.\n"
                f"STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
    except Exception as e:
        return f"An unexpected error occurred while running command: {str(e)}"

@tool
def run_python_script(script_path: str) -> Dict[str, Any]:
    """
    Executes a Python script located at `file_path` using a new Python subprocess.
    The script is executed in the current working directory ('.').
    Captures stdout, stderr, and detects any new files (e.g., plots, data files) created
    in the current directory during execution.

    Args:
        script_path (str): The path to the Python script to execute, relative to the current directory ('.').

    Returns:
        Dict[str, Any]: A dictionary containing:
                        - 'status': 'success' or 'error'
                        - 'stdout': Standard output from the script.
                        - 'stderr': Standard error from the script.
                        - 'created_files': List of paths to new files created by the script (relative to '.').
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
            ["python", script_path], # No need for basename or changing dir
            capture_output=True,
            text=True,
            timeout=180 # Timeout after 180 seconds (3 minutes)
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

if __name__ == '__main__':
    # Define test filenames and content
    test_txt_filename = "test_file.txt"
    test_py_filename = "test_script.py"
    test_py_output_filename = "output_from_script.txt"

    test_txt_content = "Hello from the test suite!"
    test_py_content = (
        'import sys\n'
        'print("This is stdout from the test script.")\n'
        'print("This is stderr from the test script.", file=sys.stderr)\n'
        'with open("output_from_script.txt", "w") as f:\n'
        '    f.write("This file was created by the python script.")\n'
    )

    try:
        # === Test 1: write_file tool ===
        print("--- Testing: write_file ---")
        result_write = write_file.invoke({"file_path": test_txt_filename, "content": test_txt_content})
        print(result_write)
        assert os.path.exists(test_txt_filename), "write_file did not create the file!"
        with open(test_txt_filename, "r") as f:
            assert f.read() == test_txt_content, "write_file did not write the correct content!"
        print("✅ write_file PASSED\n")

        # === Test 2: read_file tool ===
        print("--- Testing: read_file ---")
        result_read = read_file.invoke({"file_path": test_txt_filename})
        print(result_read)
        assert test_txt_content in result_read, "read_file did not return the correct content!"
        print("✅ read_file PASSED\n")

        # === Test 3: run_shell_command tool ===
        print("--- Testing: run_shell_command ---")
        result_shell = run_shell_command.invoke({"command": 'dir'})
        print(result_shell)
        print("✅ run_shell_command PASSED\n")

        # === Test 4: run_python_script tool ===
        # This is a multi-step test. First, we write the script to test.
        print("--- Testing: run_python_script ---")
        print(f"(Step 4.1: Writing '{test_py_filename}' to disk...)")
        write_file.invoke({"file_path": test_py_filename, "content": test_py_content})
        assert os.path.exists(test_py_filename), "Failed to create the test python script."
        print("(Step 4.1: Done.)")

        # Now, execute the script we just wrote
        print(f"(Step 4.2: Executing '{test_py_filename}'...)")
        # Note: Using the recommended 'script_path' argument name
        result_py = run_python_script.invoke({"script_path": test_py_filename})
        print(f"Result from run_python_script:\n{result_py}")

        # Verify all parts of the result
        assert result_py['status'] == 'success', "Python script did not run successfully."
        assert "This is stdout" in result_py['stdout'], "Did not capture stdout correctly."
        assert "This is stderr" in result_py['stderr'], "Did not capture stderr correctly."
        assert os.path.exists(test_py_output_filename), "Python script did not create its output file."
        assert len(result_py['created_files']) > 0, "Did not detect created files."
        assert test_py_output_filename in result_py['created_files'][0], "Did not report the correct created file."
        print("✅ run_python_script PASSED\n")

        print("--- All tools passed! ---")

    finally:
        # --- Cleanup: Remove all test files ---
        print("\n--- Cleaning up test files ---")
        for f in [test_txt_filename, test_py_filename, test_py_output_filename]:
            if os.path.exists(f):
                os.remove(f)
                print(f"Removed '{f}'")
        print("--- Cleanup complete ---")