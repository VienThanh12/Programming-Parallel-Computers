from ppcgrader.info_utils import *
from ppcgrader.doc_builder import *
from ppcgrader.reporter import json_to_output

code = "llm"
name = "LLM"
descr = "large language model"


def explain_web(raw: dict):
    return generate_html(explain(json_to_output(raw), "web"))


def explain_terminal(r, color=False):
    printer = TerminalPrinter(color=color)
    if color:
        # Define the highlight color for errors (Red/Bold)
        hl, reset = '\033[31;1m', '\033[0m'
        printer.set_format("strong", hl, reset)
    else:
        printer.set_format("strong", '', '')

    return generate_term(explain(r, "term"), printer)


def explain(r, mode: str):

    domain = get_domain(mode=mode, domain=code)

    builder = DocumentBuilder(mode)

    input_data = r.input_data or {}
    oe = r.output_errors or {}

    config = input_data.get('config', {})
    length = input_data.get('length', None)

    max_error = oe.get("max_error", None)
    location_pos = oe.get("location_pos", None)
    location_tok = oe.get("location_tok", None)
    threshold = oe.get("threshold", None)

    if config:
        with builder.paragraph() as txt:
            txt += domain.gettext('function_called_with_model')

        with builder.list(style="compact") as lst:
            for param, value in config.items():
                lst.add_item(f'{param} = {value}')

        if length:
            with builder.paragraph() as txt:
                txt += domain.gettext('processed_token_sequence',
                                      length=length)

    if safenum(max_error) > 0:
        with builder.paragraph() as txt:
            txt += domain.gettext('probabilities_mismatch')
            if location_pos is not None and location_tok is not None:
                txt += domain.gettext(
                    'largest_mismatch_at',
                    location_pos=location_pos,
                    location_tok=location_tok,
                )

        if max_error is not None and threshold is not None:
            with builder.paragraph() as txt:
                txt += domain.gettext('largest_error_intro')
                # Create a styled node for the error value to handle highlighting
                txt += StringNode(safereadable(max_error), style="strong")
                txt += domain.gettext('largest_error_period')

                txt += domain.gettext('largest_error_limit',
                                      threshold=safereadable(threshold))

                if safenum(threshold) > 0:
                    rel = max_error / threshold
                    txt += domain.gettext(
                        'largest_error_ratio',
                        error_ratio=safereadable(rel),
                    )

            # Checks if the ratio is relatively small (using 100 as the upper bound based on terminal logic)
            if safenum(threshold) > 0 and (max_error / threshold) < 100:
                with builder.paragraph() as txt:
                    txt += domain.gettext('errors_maybe_rounding')

    return builder.build()
