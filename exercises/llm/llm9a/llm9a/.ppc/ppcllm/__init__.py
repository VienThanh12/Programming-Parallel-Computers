import os.path
from typing import List, Optional
from ppcgrader.compiler import Compiler
import ppcgrader.config
from urllib.request import urlretrieve


class Config(ppcgrader.config.Config):
    def __init__(self, code: str):
        from . import info
        super().__init__(binary='llm',
                         cfg_file=__file__,
                         openmp=True,
                         info=info,
                         code=code)

    def demo_flags(self, compiler: Compiler) -> Compiler:
        return self.common_flags(compiler)

    def demo_command(self, args: List[str]) -> List[str]:
        if len(args) == 0 or len(args) == 1:
            # download model and tokenizer
            if not os.path.exists("stories15M.bin"):
                urlretrieve(
                    "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
                    "stories15M.bin")
            if not os.path.exists("tokenizer.bin"):
                urlretrieve(
                    "https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin",
                    "tokenizer.bin")
            text = "Once upon a time, there was a test story" if len(
                args) == 0 else args[0]
            args = ["stories15M.bin", "tokenizer.bin", text]
        return [os.path.join('./', self.demo_binary)] + args

    def _estimate_flops(self, dim: Optional[int], hidden_dim: Optional[int],
                        tokens: Optional[int], vocab: Optional[int],
                        layers: Optional[int]):
        if dim is None or hidden_dim is None or tokens is None or vocab is None or layers is None:
            return None

        rms_norm = dim * 4  # 2 for inner product, 2 for normalization
        mm_dd = 2 * dim * dim
        mm_dh = 2 * dim * hidden_dim
        att_scores = 2 * dim * tokens * (tokens + 1) // 2
        att_gather = 4 * dim  # max, exp, sum, norm
        att_lookup = 2 * dim + tokens * (tokens + 1) // 2
        swiglu = 5 * hidden_dim  # exp, div, mul, plus, minus
        residual = dim
        rope = 2 * dim
        # 1 mm for q, k, v, attproj in ATT, 3 mm_dh for FFN
        per_token = 2 * rms_norm + 4 * mm_dd + rope + att_gather + 3 * mm_dh + swiglu + 2 * residual
        per_block = per_token * tokens + att_scores + att_lookup

        # output
        out = rms_norm + dim * vocab * tokens * 2

        return per_block * layers + out

    def parse_output(self, output):
        input_data = {"seed": None, "length": None, "config": {}}
        output_data = {}
        output_errors = {
            "max_error": None,
            "location_pos": None,
            "location_tok": None,
        }
        statistics = {}

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
            elif splitted[0] in ['max_error', 'threshold']:
                output_errors[splitted[0]] = float(splitted[1])
            elif splitted[0] in ['location_tok', 'location_pos']:
                output_errors[splitted[0]] = int(splitted[1])
            elif splitted[0] in ['seed', 'length']:
                input_data[splitted[0]] = int(splitted[1])
            elif splitted[0].startswith("config."):
                input_data["config"][splitted[0][7:]] = int(splitted[1])

        flops = self._estimate_flops(
            dim=input_data["config"].get("dim", None),
            hidden_dim=input_data["config"].get("hidden_dim", None),
            layers=input_data["config"].get("n_layers", None),
            vocab=input_data["config"].get("vocab_size", None),
            tokens=input_data.get("length", None),
        )
        if flops:
            statistics['operations'] = flops
            statistics['operations_name'] = "useful arithmetic operation"

        return time, errors, input_data, output_data, output_errors, statistics
