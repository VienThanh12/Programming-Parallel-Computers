#!/usr/bin/env python3

from ppcgrader.cli import cli
import ppcmf

if __name__ == "__main__":
    cli(ppcmf.Config(code='mf1', openmp=False))
