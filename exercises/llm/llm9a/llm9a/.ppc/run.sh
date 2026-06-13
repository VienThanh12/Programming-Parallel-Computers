#!/usr/bin/env bash
set -e
cat > /box/llm.cc
chmod a-w /box/llm.cc

cd /program
/program/.ppc/grader.py --file "/box/llm.cc" --binary "/box/llm" --json "$@"
