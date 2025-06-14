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

## Running the Evaluation

### Using the Default OpenAI Model Runner

```bash
# Evaluate the default OpenAI model (gpt-3.5-turbo)
python browsecomp.py --examples 5

# Evaluate a specific OpenAI model
python browsecomp.py --model-name gpt-4 --examples 5

# Evaluate a specific OpenAI model
python browsecomp.py --model-name gpt-4 --examples 5
```

### Using a Custom Runner

You can evaluate any custom implementation by creating an executable that follows the input/output protocol:

```bash
# Using a custom implementation
python browsecomp.py --runner-path /path/to/your/runner/executable --examples 5
```

### Command Line Arguments

- `--runner-path`: Path to evaluation runner executable (default: model_runner.py)
- `--model-name`: Model name to pass to the runner
- `--examples`: Number of examples to evaluate (default: 10)

The evaluation will generate an HTML report with the results in the current directory.

## Creating a Custom Runner

To create a custom runner, you need to create an executable (script or binary) that:

1. Accepts a JSON input with a "prompt" field
2. Returns a JSON output with a "response" field

The model_runner.py script provided serves as a reference implementation. Your custom implementation must:

1. Accept input either from a file specified with `--input` or from stdin
2. Output results either to a file specified with `--output` or to stdout
3. Follow the JSON format for input/output

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
