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

```bash
python browsecomp.py --model gpt-4 --examples 5
```

Command line arguments:
- `--model`: The OpenAI model to evaluate (default: gpt-3.5-turbo)
- `--examples`: Number of examples to evaluate (default: 10)

The evaluation will generate an HTML report with the results in the current directory.

## Understanding Results

The evaluation will output:
- Accuracy score (percentage of correctly answered questions)
- Detailed metrics on correct and incorrect answers
- HTML representation of results (if enabled)

## Implementation Details

The `browsecomp.py` implementation includes:
- Encryption/decryption of test data for security
- Template-based querying of language models
- Automatic grading of responses using a grader model
- Result aggregation and reporting
