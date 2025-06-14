# BrowseComp Evaluation

> This repository is forked and modified from OpenAI's [simple evals github repository](https://github.com/openai/simple-evals) and used to complete evaluation for projects such as Agent TARS.

This repository contains the implementation of BrowseComp benchmark for evaluating language models. BrowseComp is a simple yet challenging benchmark for browsing agents, developed by Jason Wei, Zhiqing Sun, Spencer Papay, and others.

## About BrowseComp

[BrowseComp](https://openai.com/index/browsecomp/) is designed to evaluate the browsing capabilities of language models. It tests the model's ability to understand web content, answer questions accurately, and provide confidence scores for its answers.

## Setup Requirements

```bash
# Install dependencies
pip install -r requirements.txt
```

You also need to set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

## Virtual Environment Setup

```bash
# For Windows
# Create a virtual environment
python -m venv venv
# Activate the virtual environment
venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
# When you're done, deactivate the environment
deactivate

# For macOS/Linux
# Create a virtual environment
python -m venv venv
# Activate the virtual environment
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
# When you're done, deactivate the environment
deactivate

# Using Conda (cross-platform)
# Create a conda environment
conda create -n browsecomp python=3.10
# Activate the conda environment
conda activate browsecomp
# Install dependencies
pip install -r requirements.txt
# When you're done, deactivate the environment
conda deactivate
```

## Running the Evaluation

### Using the Default OpenAI Model Runner

```bash
# Evaluate a specific OpenAI model
python browsecomp.py --model-name gpt-4 --examples 5
```

### Using a Custom Python Script Runner

```bash
# Using a custom Python script implementation
python browsecomp.py --python-script /path/to/your/script.py --model-name gpt-4-turbo --examples 5
```

### Using a CLI Command

```bash
# Using a custom CLI command
python browsecomp.py --command "your-cli-tool run" --model-name your-model-id --examples 5
```

### Command Line Arguments

- `--python-script`: Path to Python script runner (default: model_runner.py)
- `--command`: CLI command format string (e.g., "agent-tars run")
- `--model-name`: Model name to pass to the runner (optional, required for certain configurations)
- `--examples`: Number of examples to evaluate (default: 10)

**Note:** The `--python-script` and `--command` are mutually exclusive execution modes:
- `--python-script`: Executes a Python script with your Python interpreter
- `--command`: Executes a shell command directly, useful for compiled programs or complex CLI tools

The evaluation will generate an HTML report with the results in the current directory.

## Creating a Custom Runner

To create a custom runner, you need to create either:

1. A Python script that:
   - Accepts an `--input` argument with the prompt text
   - Outputs the model's response to stdout
   
2. Or a CLI command that:
   - Accepts `--input "prompt"` and optionally `--model "model_name"` parameters
   - Outputs the model's response to stdout

### Python Script Example

See `model_runner.py` for a reference implementation of a Python script runner.

### CLI Command Example

Your CLI tool should accept arguments in this format:
```bash
your-cli-tool run --model "model-name" --input "prompt text here"
```

## Understanding Results

The evaluation will output:
- Accuracy score (percentage of correctly answered questions)
- Detailed metrics on correct and incorrect answers
- HTML representation of results

## Implementation Details

The `browsecomp.py` implementation includes:
- Encryption/decryption of test data for security
- Template-based querying of language models
- Automatic grading of responses using a grader model
- Result aggregation and reporting