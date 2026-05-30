import os
from typing import Optional
import ppcgrader.config


class Config(ppcgrader.config.Config):
    def __init__(self, openmp: bool, code: str):
        from . import info
        super().__init__(binary='mf',
                         cfg_file=__file__,
                         gpu=False,
                         openmp=openmp,
                         info=info,
                         code=code)
        self.default_demo = [os.path.join(self.base_dir, 'demo.png')]
        self.demo_flags = self._demo_flags_png
        self.demo_post = self._demo_post_png

    def parse_output(self, output):
        input_data = {
            "nx": None,
            "ny": None,
            "hy": None,
            "hx": None,
            "data": None,
        }
        output_data = {
            "result": None,
        }
        output_errors = {
            "locations": None,
        }
        statistics = {}

        def parse_matrix(string):
            M = string.strip('[]').split(';')
            M = [row.strip() for row in M]
            M = [row.split(" ") for row in M]
            M = [[float(e) for e in row] for row in M]
            return M

        for line in output.splitlines():
            splitted = line.split('\t')
            if splitted[0] == 'result':
                errors = {
                    'fail': True,
                    'pass': False,
                    'done': False
                }[splitted[1]]
            elif splitted[0] == 'time':
                time = float(splitted[1])
            elif splitted[0] == 'perf_wall_clock_ns':
                time = int(splitted[1]) / 1e9
                statistics[splitted[0]] = int(splitted[1])
            elif splitted[0].startswith('perf_'):
                statistics[splitted[0]] = int(splitted[1])
            elif splitted[0] in ['ny', 'nx', 'hy', 'hx']:
                input_data[splitted[0]] = int(splitted[1])
            elif splitted[0] == 'input':
                input_data["data"] = parse_matrix(splitted[1])
            elif splitted[0] == 'output':
                output_data["result"] = parse_matrix(splitted[1])
            elif splitted[0] == 'locations':
                output_errors["locations"] = parse_matrix(splitted[1])

        return time, errors, input_data, output_data, output_errors, statistics
