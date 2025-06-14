import base64
import hashlib
import json
import random
import re
import pandas
import os
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Literal, Optional, Union
from dataclasses import dataclass, field
import jinja2
import numpy as np
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import openai
from openai import OpenAI

# Types definitions (from types.py)
Message = dict[str, Any]  # keys role, content
MessageList = list[Message]

@dataclass
class SamplerResponse:
    """Response from a sampler."""
    response_text: str
    actual_queried_message_list: MessageList
    response_metadata: dict[str, Any]

class SamplerBase:
    """Base class for defining a sampling model."""
    def __call__(self, message_list: MessageList) -> SamplerResponse:
        raise NotImplementedError
    
    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

@dataclass
class SingleEvalResult:
    """Result of evaluating a single sample"""
    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    html: str | None = None
    convo: MessageList | None = None
    example_level_metadata: dict[str, Any] | None = None

@dataclass
class EvalResult:
    """Result of running an evaluation (usually consisting of many samples)"""
    score: float | None
    metrics: dict[str, float] | None
    htmls: list[str]
    convos: list[MessageList]
    metadata: dict[str, Any] | None

class Eval:
    """Base class for defining an evaluation."""
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        raise NotImplementedError

# Helper functions from common.py
def _compute_stat(values: list, stat: str):
    if stat == "mean":
        return np.mean(values)
    elif stat == "std":
        return np.std(values)
    elif stat == "min":
        return np.min(values)
    elif stat == "max":
        return np.max(values)
    elif stat == "n_samples":
        return len(values)
    elif stat == "bootstrap_std":
        return np.std(
            [np.mean(np.random.choice(values, len(values))) for _ in range(1000)]
        )
    else:
        raise ValueError(f"Unknown {stat =}")

def aggregate_results(
    single_eval_results: list[SingleEvalResult],
    default_stats: tuple[str, ...] = ("mean", "std"),
    name2stats: dict[str, tuple[str]] | None = None,
) -> EvalResult:
    """Aggregate results from multiple evaluations into a single EvalResult."""
    name2stats = name2stats or {}
    name2values = defaultdict(list)
    htmls = []
    convos = []
    metadata = []
    for single_eval_result in single_eval_results:
        for name, value in single_eval_result.metrics.items():
            name2values[name].append(value)
        if single_eval_result.score is not None:
            name2values["score"].append(single_eval_result.score)
        htmls.append(single_eval_result.html)
        convos.append(single_eval_result.convo)
        metadata.append(single_eval_result.example_level_metadata)
    final_metrics = {}
    for name, values in name2values.items():
        stats = name2stats.get(name, default_stats)
        for stat in stats:
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_stat(values, stat)
    return EvalResult(
        score=final_metrics.pop("score", None),
        metrics=final_metrics,
        htmls=htmls,
        convos=convos,
        metadata={"example_level_metadata": metadata},
    )

def map_with_progress(
    f: callable,
    xs: list[Any],
    num_threads: int = os.cpu_count() or 10,
    pbar: bool = True,
):
    """Apply f to each element of xs, using a ThreadPool, and show progress."""
    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x

    if os.getenv("debug"):
        return list(map(f, pbar_fn(xs, total=len(xs))))
    else:
        with ThreadPoolExecutor(min(num_threads, len(xs))) as executor:
            futures = {executor.submit(f, x): x for x in xs}
            results = []
            for future in pbar_fn(futures, total=len(xs)):
                # 获取 future 的实际结果，而不是输入参数
                results.append(future.result())
            return results

# Jinja setup for HTML rendering
jinja_env = jinja2.Environment(
    loader=jinja2.BaseLoader(),
    undefined=jinja2.StrictUndefined,
    autoescape=jinja2.select_autoescape(["html", "xml"]),
)

_message_template = """
<div class="message {{ role }}">
    <div class="role">
    {{ role }}
    {% if variant %}<span class="variant">({{ variant }})</span>{% endif %}
    </div>
    <div class="content">
    <pre>{{ content }}</pre>
    </div>
</div>
"""

def message_to_html(message: Message) -> str:
    """Generate HTML snippet (inside a <div>) for a message."""
    return jinja_env.from_string(_message_template).render(
        role=message["role"],
        content=message["content"],
        variant=message.get("variant", None),
    )

jinja_env.globals["message_to_html"] = message_to_html

HTML_JINJA = """
<h3>Prompt conversation</h3>
{% for message in prompt_messages %}
{{ message_to_html(message) | safe }}
{% endfor %}
<h3>Sampled message</h3>
{{ message_to_html(next_message) | safe }}
<h3>Results</h3>
<p>Correct Answer: {{ correct_answer }}</p>
<p>Extracted Answer: {{ extracted_answer }}</p>
<p>Score: {{ score }}</p>
"""

_report_template = """<!DOCTYPE html>
<html>
    <head>
        <style>
            .message {
                padding: 8px 16px;
                margin-bottom: 8px;
                border-radius: 4px;
            }
            .message.user {
                background-color: #B2DFDB;
                color: #00695C;
            }
            .message.assistant {
                background-color: #B39DDB;
                color: #4527A0;
            }
            .message.system {
                background-color: #EEEEEE;
                color: #212121;
            }
            .role {
                font-weight: bold;
                margin-bottom: 4px;
            }
            .variant {
                color: #795548;
            }
            table, th, td {
                border: 1px solid black;
            }
            pre {
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
    {% if metrics %}
    <h1>Metrics</h1>
    <table>
    <tr>
        <th>Metric</th>
        <th>Value</th>
    </tr>
    <tr>
        <td><b>Score</b></td>
        <td>{{ score | float | round(3) }}</td>
    </tr>
    {% for name, value in metrics.items() %}
    <tr>
        <td>{{ name }}</td>
        <td>{{ value }}</td>
    </tr>
    {% endfor %}
    </table>
    {% endif %}
    <h1>Examples</h1>
    {% for html in htmls %}
    {{ html | safe }}
    <hr>
    {% endfor %}
    </body>
</html>
"""

def make_report(eval_result: EvalResult) -> str:
    """Create a standalone HTML report from an EvalResult."""
    return jinja_env.from_string(_report_template).render(
        score=eval_result.score,
        metrics=eval_result.metrics,
        htmls=eval_result.htmls,
    )

# ChatCompletionSampler implementation for grader model
class ChatCompletionSampler(SamplerBase):
    """Sample from OpenAI's chat completion API"""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = "You are a helpful assistant.",
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.client = OpenAI()
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI API returned empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1

# New ExternalProcessSampler implementation
class ExternalProcessSampler(SamplerBase):
    """Sample from an external executable process."""

    def __init__(
        self,
        executable_path: str,
        system_message: str | None = None,
        model_name: str | None = None,
        timeout: int = 60,
    ):
        self.executable_path = executable_path
        self.system_message = system_message
        self.model_name = model_name
        self.timeout = timeout

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        # Prepare input data
        full_messages = message_list
        if self.system_message:
            full_messages = [
                self._pack_message("system", self.system_message)
            ] + message_list
        
        # Combine messages into a single prompt for simplicity
        prompt = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in full_messages])
        
        try:
            import sys
            python_executable = sys.executable
            print(f"Using python executable: {python_executable}")

            # Run external process directly with command line arguments
            cmd = [python_executable, self.executable_path, "--input", prompt]
            if self.model_name:
                cmd.extend(["--model", self.model_name])
                
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = process.communicate(timeout=self.timeout)
            
            # Use stdout directly as response text
            response_text = stdout.decode('utf-8')
            
            if process.returncode != 0:
                print(f"Error from external process: {stderr.decode()}")
                response_text = f"Error: Process returned code {process.returncode}"
            
            return SamplerResponse(
                response_text=response_text,
                response_metadata={},
                actual_queried_message_list=full_messages,
            )
            
        except subprocess.TimeoutExpired:
            process.kill()
            return SamplerResponse(
                response_text="Error: Process timeout",
                response_metadata={"error": "timeout"},
                actual_queried_message_list=full_messages,
            )
            
        except Exception as e:
            print(f"Error running external process: {e}")
            return SamplerResponse(
                response_text=f"Error: {str(e)}",
                response_metadata={"error": str(e)},
                actual_queried_message_list=full_messages,
            )

# Import required for defaultdict
from collections import defaultdict

# BrowseComp evaluation implementation
QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
""".strip()

CHOICE_STRINGS = ["yes", "no"]

def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]

def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()

class BrowseCompEval(Eval):
    def __init__(self, grader_model: SamplerBase, num_examples: int | None = None, n_repeats: int = 1):
        df = pandas.read_csv(
            "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
        )
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported when max_examples = None"
            rng = random.Random(0)
            examples = rng.sample(examples, num_examples)
        self.examples = examples * n_repeats
        self.grader_model = grader_model

    def grade_sample(self, question: str, correct_answer: str, response: str) -> str:
        grader_prompt = GRADER_TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            response=response,
        )

        prompt_messages = [
            self.grader_model._pack_message(content=grader_prompt, role="user")
        ]
        sampler_response = self.grader_model(prompt_messages)
        grading_response = sampler_response.response_text

        match = re.search(r"correct: (yes|no)", grading_response)
        return match.group(1).lower() if match else "no"  # Default to "no" if no match

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            problem = decrypt(row.get("problem", ""), row.get("canary", ""))
            answer = decrypt(row.get("answer", ""), row.get("canary", ""))
            prompt_messages = [
                sampler._pack_message(content=QUERY_TEMPLATE.format(Question=problem), role="user")
            ]
            sampler_response = sampler(prompt_messages)
            response_text = sampler_response.response_text
            actual_queried_prompt_messages = sampler_response.actual_queried_message_list
            grade_result = self.grade_sample(problem, answer, response_text)

            # Metrics based on grading response
            is_correct = grade_result == "yes"
            is_incorrect = grade_result == "no"
            
            score = is_correct

            # Create HTML for each sample result
            html = jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=actual_queried_prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=answer,
                extracted_answer=response_text,
            )
            convo = actual_queried_prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(html=html, score=score, convo=convo, metrics={
                "is_correct": is_correct,
                "is_incorrect": is_incorrect,
            })

        # Run evaluation and collect results
        results = map_with_progress(fn, self.examples)

        # Aggregate metrics
        aggregate_metrics = {
            "is_correct": sum(result.metrics["is_correct"] for result in results) / len(results),
            "is_incorrect": sum(result.metrics["is_incorrect"] for result in results) / len(results),
        }
        print("AGGREGATE METRICS") 
        print(aggregate_metrics) 
        print("##################")

        output_d = {
            "accuracy": aggregate_metrics["is_correct"],
        }
        
        print(f"Accuracy: {output_d['accuracy']:.3f}")
        
        return aggregate_results(results)

# Example usage
def run_browsecomp_eval(runner_path="model_runner.py", model_name=None, num_examples=10):
    # Set up the model to evaluate - either external process or OpenAI API
    if os.path.isfile(runner_path):
        # Make sure the model runner has execute permissions
        if runner_path.endswith('.py'):
            os.chmod(runner_path, 0o755)
        
        # Use the external process sampler
        print(f"Using external runner: {runner_path}")
        model = ExternalProcessSampler(
            executable_path=runner_path,
            model_name=model_name
        )
    else:
        # Fallback to OpenAI API
        print(f"Using OpenAI API with model: {model_name or 'gpt-3.5-turbo'}")
        model = ChatCompletionSampler(model=model_name or "gpt-3.5-turbo")
    
    # Set up the grader model
    grader_model = ChatCompletionSampler(model="gpt-4")
    
    # Initialize the evaluation
    eval = BrowseCompEval(grader_model=grader_model, num_examples=num_examples)
    
    # Run the evaluation
    result = eval(model)
    
    # Generate report
    report = make_report(result)
    
    # Create a filename for the report
    model_id = model_name or os.path.basename(runner_path).replace('.py', '')
    report_filename = f"browsecomp_{model_id}_{num_examples}_examples.html"
    
    # Save report
    with open(report_filename, "w") as f:
        f.write(report)
    
    print(f"Evaluation complete! Report saved to {report_filename}")
    return result

if __name__ == "__main__":
    # Example usage: python browsecomp.py
    import argparse
    
    parser = argparse.ArgumentParser(description="Run BrowseComp evaluation")
    parser.add_argument("--runner-path", type=str, default="model_runner.py", 
                        help="Path to evaluation runner executable/script")
    parser.add_argument("--model-name", type=str, help="Model name to pass to runner")
    parser.add_argument("--examples", type=int, default=10, help="Number of examples to evaluate")
    args = parser.parse_args()
    
    run_browsecomp_eval(runner_path=args.runner_path, model_name=args.model_name, num_examples=args.examples)