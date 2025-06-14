#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import sys
import os
import openai
from openai import OpenAI

def setup_openai_client():
    """Setup and return OpenAI client."""
    return OpenAI()

def generate_response(prompt, model_name):
    """Generate response using OpenAI API."""
    client = setup_openai_client()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=1024,
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content
    except Exception as e:
        return "Error generating response: {}".format(str(e))

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Model runner for BrowseComp evaluation")
    parser.add_argument("--input", type=str, help="Query string to process", required=True)
    parser.add_argument("--output", type=str, help="Output response file path (optional, defaults to stdout)")
    parser.add_argument("--model", type=str, help="Model name to use (default: gpt-4)", default="gpt-4")
    args = parser.parse_args()
    
    # Use direct query string from input argument
    prompt = args.input
    model = args.model
    
    # Generate response
    result = generate_response(prompt, model)
    
    # Output result - either to file or stdout
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
    else:
        sys.stdout.write(result)
        sys.stdout.flush()

if __name__ == "__main__":
    main()
