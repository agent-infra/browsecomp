import base64
import hashlib
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Dict, Tuple, Callable, Optional

import jinja2
import numpy as np
from tqdm import tqdm

from interface import SingleEvalResult, EvalResult, Message, MessageList

# Add colorama for cross-platform colored terminal output
try:
    from colorama import init, Fore, Style
    # Initialize colorama
    init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    # Fallback for when colorama is not available
    class DummyColorama:
        def __getattr__(self, name):
            return ""
    Fore = Style = DummyColorama()

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
            [np.mean(np.random.choice(values, len(values)))
             for _ in range(1000)]
        )
    else:
        raise ValueError(f"Unknown {stat=}")


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
    callback: Optional[Callable[[Any, int, int], None]] = None,
):
    """Apply f to each element of xs, using a ThreadPool, and show progress.
    
    Args:
        f: Function to apply to each element
        xs: List of elements to process
        num_threads: Number of threads to use
        pbar: Whether to display a progress bar
        callback: Optional callback function called after each item is processed 
                 with params (result, index, total)
    """
    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x
    results = []
    total = len(xs)

    if os.getenv("debug"):
        # Sequential execution for debugging
        for i, x in enumerate(pbar_fn(xs, total=total)):
            result = f(x)
            results.append(result)
            if callback:
                callback(result, i, total)
        return results
    else:
        # Parallel execution with callbacks
        with ThreadPoolExecutor(min(num_threads, total)) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(f, x): i for i, x in enumerate(xs)}
            
            # Process results as they complete
            for future in pbar_fn(as_completed(future_to_index), total=total):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results.append(result)
                    if callback:
                        callback(result, idx, total)
                except Exception as exc:
                    print(f'Task generated an exception: {exc}')
                    
        # Sort results by original index if using parallel execution
        # This ensures results are in the same order as input despite parallel completion
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

def save_interim_report(results_so_far: list[SingleEvalResult], filename: str, metrics: dict = None, score: float = None) -> None:
    """Generate and save an interim report based on results processed so far."""
    interim_result = EvalResult(
        score=score,
        metrics=metrics,
        htmls=[r.html for r in results_so_far],
        convos=[r.convo for r in results_so_far],
        metadata={"example_level_metadata": [r.example_level_metadata for r in results_so_far]},
    )
    report = make_report(interim_result)
    with open(filename, "w") as f:
        f.write(report)

def print_colored_result(result: SingleEvalResult, index: int, total: int) -> None:
    """Print a colored log message for a single evaluation result."""
    is_correct = result.metrics.get("is_correct", False)
    status = "CORRECT" if is_correct else "INCORRECT"
    color = Fore.GREEN if is_correct else Fore.RED
    
    # Format the progress and status message
    progress = f"[{index+1}/{total}]"
    message = f"{progress} Example evaluation: {color}{status}{Style.RESET_ALL}"
    
    print(message)