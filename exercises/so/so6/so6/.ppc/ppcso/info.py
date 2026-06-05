from ppcgrader.info_utils import *
from ppcgrader.doc_builder import *
from ppcgrader.reporter import json_to_output

code = "so"
name = "SO"
descr = "sorting"


def explain_web(raw: dict):
    return generate_html(explain(json_to_output(raw), "web"))


def explain_terminal(r, color=False):
    printer = TerminalPrinter(color=color)
    return generate_term(explain(r, "term"), printer)


def explain(r, mode: str):

    domain = get_domain(mode=mode, domain=code)

    fail_style = "verywrong"
    if mode == "web":
        fail_style = "strong"  # preserve old web emphasis

    builder = DocumentBuilder(mode)
    input = r.input_data or {}
    output = r.output_data or {}
    oe = r.output_errors or {}

    n = input.get("n", None)
    data = input.get("data", None)
    result = output.get("result", None)
    correct = oe.get("correct", None)
    error_type = oe.get("type", None)

    if data is not None and result is not None and correct is not None and n is not None:

        with builder.paragraph() as txt:
            txt += domain.gettext("function_called_with_n", n=n)

        with builder.paragraph() as txt:
            txt += domain.gettext("input_output_expected_intro")

        if mode == "term":  #Hack to fix the terminal output to match the old format
            with builder.text() as txt:
                txt += domain.gettext("term_header_row")
                for i in range(n):
                    x = safeprint(safeget(data, i))
                    y = safeprint(safeget(result, i))
                    z = safeprint(safeget(correct, i))

                    txt += f"{x:>20s} "
                    if y != z:
                        txt += StringNode(f"{y:>20s}", fail_style)
                        txt += " "
                        txt += StringNode(f"{z:>20s}", fail_style)
                    else:
                        txt += f"{y:>20s} {z:>20s}"
                    txt += "\n"
        else:
            with builder.table(n, 3, aligns=["right", "right",
                                             "right"]) as tbl:
                tbl.header(0, domain.gettext("input_header"))
                tbl.header(1, domain.gettext("output_header"))
                tbl.header(2, domain.gettext("expected_header"))

                for i in range(n):
                    x = safeget(data, i)
                    y = safeget(result, i)
                    z = safeget(correct, i)

                    tbl.entry(i, 0, safeprint(x))
                    if y != z:
                        tbl.entry(i, 1, StringNode(safeprint(y), fail_style))
                        tbl.entry(i, 2, StringNode(safeprint(z), fail_style))
                    else:
                        tbl.entry(i, 1, safeprint(y))
                        tbl.entry(i, 2, safeprint(z))

    if error_type == 1:
        with builder.paragraph() as txt:
            txt += domain.gettext("output_not_in_order")
    elif error_type == 2:
        with builder.paragraph() as txt:
            txt += domain.gettext("output_wrong_numbers")

    return builder.build()
