import re
import random
import time
from typing import Dict, List, Optional

import pandas

from interface import SamplerBase, SingleEvalResult, EvalResult, Eval
from utils import decrypt, aggregate_results, map_with_progress, jinja_env, HTML_JINJA
from utils import print_colored_result, save_interim_report

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


class BrowseCompEval(Eval):
    def __init__(
        self, 
        grader_model: SamplerBase, 
        num_examples: int | None = None, 
        n_repeats: int = 1,
        exclude_keywords: list[str] = None
    ):
        df = pandas.read_csv(
            "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
        )
        examples = [row.to_dict() for _, row in df.iterrows()]

        # Filter examples based on exclude_keywords if provided
        if exclude_keywords:
            filtered_examples = []
            for example in examples:
                # Decrypt problem to check if it contains exclude keywords
                problem = decrypt(example.get("problem", ""),
                                example.get("canary", ""))
                if not any(keyword.lower() in problem.lower() for keyword in exclude_keywords):
                    filtered_examples.append(example)

            print(
                f"Excluded {len(examples) - len(filtered_examples)} examples containing keywords: {exclude_keywords}")
            examples = filtered_examples

        if num_examples:
            assert n_repeats == 1, "n_repeats only supported when max_examples = None"
            rng = random.Random(0)
            examples = rng.sample(examples, min(num_examples, len(examples)))

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
        # Default to "no" if no match
        return match.group(1).lower() if match else "no"

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            problem = decrypt(row.get("problem", ""), row.get("canary", ""))
            answer = decrypt(row.get("answer", ""), row.get("canary", ""))
            prompt_messages = [
                sampler._pack_message(content=QUERY_TEMPLATE.format(
                    Question=problem), role="user")
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
            convo = actual_queried_prompt_messages + \
                [dict(content=response_text, role="assistant")]
            return SingleEvalResult(html=html, score=score, convo=convo, metrics={
                "is_correct": is_correct,
                "is_incorrect": is_incorrect,
            })

        # List to track processed results
        processed_results = []
        
        # Generate the report filename for interim updates
        model_id = getattr(sampler, 'model', 'default')
        if hasattr(sampler, 'model_name') and sampler.model_name:
            model_id = sampler.model_name
        interim_report_filename = f"browsecomp_{model_id}_{len(self.examples)}_examples_in_progress.html"
        
        # Callback function for processing each result
        def process_result(result, index, total):
            # Add the result to our processed list
            processed_results.append(result)
            
            # Calculate interim metrics
            correct_count = sum(1 for r in processed_results if r.metrics["is_correct"])
            accuracy = correct_count / len(processed_results)
            interim_metrics = {
                "is_correct": accuracy,
                "is_incorrect": 1 - accuracy,
                "examples_completed": len(processed_results),
                "total_examples": total,
            }
            
            # Print colored status
            print_colored_result(result, index, total)
            
            # Save interim report
            save_interim_report(
                processed_results,
                interim_report_filename,
                metrics=interim_metrics,
                score=accuracy
            )
            
            # Optionally print summary stats periodically (e.g., every 5 examples)
            if len(processed_results) % 5 == 0 or len(processed_results) == total:
                print(f"\nINTERIM METRICS (after {len(processed_results)}/{total} examples)")
                print(f"Current Accuracy: {accuracy:.3f}")
                print(f"Interim report updated: {interim_report_filename}")
                print("="*50)

        # Run evaluation with callback for real-time processing
        results = map_with_progress(fn, self.examples, callback=process_result)

        # Aggregate metrics (final)
        aggregate_metrics = {
            "is_correct": sum(result.metrics["is_correct"] for result in results) / len(results),
            "is_incorrect": sum(result.metrics["is_incorrect"] for result in results) / len(results),
        }
        print("\nFINAL METRICS")
        print(aggregate_metrics)
        print("##################")

        output_d = {
            "accuracy": aggregate_metrics["is_correct"],
        }

        print(f"Final Accuracy: {output_d['accuracy']:.3f}")

        return aggregate_results(results)


def run_browsecomp_eval(
    runner_path="model_runner.py",
    model_name=None,
    num_examples=10,
    cli_format=None,
    grader_model_name="gpt-4",
    grader_api_key=None,
    grader_base_url=None,
    exclude_keywords=None
):
    from samplers import ChatCompletionSampler, ExternalProcessSampler
    from utils import make_report
    import os

    # Set up the model to evaluate - can be external process or OpenAI API
    if cli_format:
        # 1. CLI Command Mode (highest priority)
        print(f"Using CLI command format: {cli_format}")
        model = ExternalProcessSampler(
            executable_path=None,
            model_name=model_name,
            cli_format=cli_format
        )
    elif os.path.isfile(runner_path):
        # 2. Python Script Mode
        # Make sure the model runner has execute permissions
        if runner_path.endswith('.py'):
            os.chmod(runner_path, 0o755)

        # Check if script requires model_name parameter
        with open(runner_path, "r") as f:
            script_content = f.read()
            requires_model = "--model" in script_content and "required=True" in script_content

        if requires_model and not model_name:
            raise ValueError(
                f"model_name must be specified for this runner: {runner_path}")

        print(f"Using external runner: {runner_path}")
        model = ExternalProcessSampler(
            executable_path=runner_path,
            model_name=model_name
        )
    else:
        # 3. OpenAI API Mode (fallback)
        if not model_name:
            raise ValueError(
                "model_name must be specified when using OpenAI API")

        print(f"Using OpenAI API with model: {model_name}")
        model = ChatCompletionSampler(model=model_name)

    # Set up grader model with custom API parameters
    grader_model = ChatCompletionSampler(
        model=grader_model_name,
        api_key=grader_api_key,
        base_url=grader_base_url
    )

    # Initialize evaluation with exclude_keywords parameter
    eval = BrowseCompEval(
        grader_model=grader_model,
        num_examples=num_examples,
        exclude_keywords=exclude_keywords
    )

    # Run evaluation
    start_time = time.time()
    print(f"Starting evaluation with {num_examples} examples...")
    result = eval(model)
    end_time = time.time()
    evaluation_time = end_time - start_time

    # Generate final report
    report = make_report(result)

    # Create report filename
    model_id = model_name if model_name else "default"
    report_filename = f"browsecomp_{model_id}_{num_examples}_examples.html"

    # Save final report
    with open(report_filename, "w") as f:
        f.write(report)

    print(f"Evaluation complete in {evaluation_time:.2f} seconds!")
    print(f"Final report saved to {report_filename}")
    return result