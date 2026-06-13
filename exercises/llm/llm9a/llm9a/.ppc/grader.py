#!/usr/bin/env python3

from ppcgrader.cli import cli
import ppcllm

if __name__ == "__main__":
    cli(ppcllm.Config(code='llm9a'))
