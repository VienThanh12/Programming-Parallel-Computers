from ppcgrader.info_utils import *
from ppcgrader.doc_builder import *
from ppcgrader.reporter import json_to_output

code = "mf"
name = "MF"
descr = "median filter"


def explain_web(raw: dict):
    return generate_html(explain(json_to_output(raw), "web"))


def explain_terminal(r, color=False):
    printer = TerminalPrinter(color=color)
    return generate_term(explain(r, "term"), printer)


def explain(r, mode: str):
    fail_style = "verywrong"
    error1, error2 = '\033[31;1m', '\033[0m'

    if mode == "web":
        fail_style = "strong"  # preserve old web format

    domain = get_domain(mode=mode, domain=code)

    builder = DocumentBuilder(mode)
    input = r.input_data or {}
    output = r.output_data or {}
    oe = r.output_errors or {}

    ny = input.get("ny", None)
    nx = input.get("nx", None)
    hy = input.get("hy", None)
    hx = input.get("hx", None)
    data = input.get("data", None)
    result = output.get("result", None)
    locations = oe.get("locations", None)

    if ny is not None and nx is not None and hy is not None and hx is not None:
        with builder.paragraph() as txt:
            txt += domain.gettext("function_called_with_params")
        with builder.list() as lst:
            with lst.item() as item:
                item += f"ny = {ny}"
            with lst.item() as item:
                item += f"nx = {nx}"
            with lst.item() as item:
                item += f"hy = {hy}"
            with lst.item() as item:
                item += f"hx = {hx}"

        if data is not None:
            with builder.paragraph() as txt:
                txt += domain.gettext("input_data_like")
            with builder.matrix(ny, nx) as mat:
                for y in range(ny):
                    for x in range(nx):
                        mat.entry(y, x, safeprint(safeget(data, y, x)))

        if result is not None:
            with builder.paragraph() as txt:
                txt += domain.gettext("output")
            with builder.matrix(ny, nx) as mat:
                for y in range(ny):
                    for x in range(nx):
                        v = safeget(result, y, x)
                        if locations and safeget(locations, y, x) == 1:
                            mat.entry(y, x, StringNode(safeprint(v),
                                                       fail_style))
                        else:
                            mat.entry(y, x, safeprint(v))

            if locations is not None:
                with builder.paragraph() as txt:
                    txt += domain.gettext("cells_wrong_highlighted_prefix")
                    txt += strong(domain.gettext("highlighted"))
                    txt += domain.gettext("cells_wrong_highlighted_suffix")

    return builder.build()
