import os
import shlex
import subprocess
import time
from typing import Optional

import openai
from openai import OpenAI

from interface import SamplerBase, SamplerResponse, MessageList


class ChatCompletionSampler(SamplerBase):
    """Sample from OpenAI's chat completion API"""

    def __init__(
        self,
        model: str,
        system_message: str | None = "You are a helpful assistant.",
        temperature: float = 0.5,
        max_tokens: int = 1024,
        api_key: str | None = None,
        base_url: str | None = None,
    ):

        # Initialize OpenAI client with custom parameters if provided
        if api_key or base_url:
            self.client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=base_url
            )
        else:
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
                    raise ValueError(
                        "OpenAI API returned empty response; retrying")
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


class ExternalProcessSampler(SamplerBase):
    """Sample from an external executable process."""

    def __init__(
        self,
        executable_path: str | None,
        system_message: str | None = None,
        model_name: str | None = None,
        cli_format: str | None = None,
    ):
        self.executable_path = executable_path
        self.system_message = system_message
        self.model_name = model_name
        self.cli_format = cli_format  # CLI command format, e.g. "agent-tars run"

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        # Prepare input data
        full_messages = message_list
        if self.system_message:
            full_messages = [
                self._pack_message("system", self.system_message)
            ] + message_list

        # Combine messages into a single prompt for simplicity
        prompt = "\n\n".join(
            [f"{msg['role'].upper()}: {msg['content']}" for msg in full_messages])

        try:
            # Properly escape the prompt to handle quotes
            escaped_prompt = prompt.replace('"', '\\"')

            # Decide command line construction based on CLI format
            if self.cli_format:
                # Use CLI command format
                cmd_parts = shlex.split(self.cli_format)
                # Add model parameter (if provided)
                if self.model_name:
                    cmd_parts.extend(["--model", self.model_name])
                # Add input parameter with proper quoting
                cmd_parts.append("--input")
                cmd_parts.append(f'"{escaped_prompt}"')

                # For shell execution, join with spaces
                cmd = " ".join(cmd_parts)
                print(f"Running CLI command: {cmd}")

                # Use shell=True to properly handle the quoted input
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True
                )
            else:
                # Original Python script execution method
                import sys
                python_executable = sys.executable
                print(f"Using python executable: {python_executable}")

                # Use list format for arguments (subprocess handles escaping)
                cmd = [python_executable,
                       self.executable_path, "--input", prompt]
                if self.model_name:
                    cmd.extend(["--model", self.model_name])

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

            stdout, stderr = process.communicate()

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

        except Exception as e:
            print(f"Error running external process: {e}")
            return SamplerResponse(
                response_text=f"Error: {str(e)}",
                response_metadata={"error": str(e)},
                actual_queried_message_list=full_messages,
            )
