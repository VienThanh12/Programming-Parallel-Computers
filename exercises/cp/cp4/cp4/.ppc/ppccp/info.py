from ppcgrader.info_utils import *
from ppcgrader.doc_builder import *
from ppcgrader.reporter import json_to_output

code = "cp"
name = "CP"
descr = "correlated pairs"


def explain_web(raw: dict):
    return generate_html(explain(json_to_output(raw), "web"))


def explain_terminal(r, color=False):
    printer = TerminalPrinter(color=color)
    if color:
        hl, minor, reset = '\033[31;1m', '\033[34;1m', '\033[0m'
        printer.set_format("element slightlywrong", minor, reset)
        printer.set_format("element verywrong", hl, reset)
    else:
        printer.set_format("element slightlywrong", '(', ')')
        printer.set_format("element verywrong", '[', ']')

    return generate_term(explain(r, "term"), printer)


def explain(r, mode: str):

    domain = get_domain(mode=mode, domain=code)

    builder = DocumentBuilder(mode)
    input = r.input_data or {}
    output = r.output_data or {}
    oe = r.output_errors or {}

    ny = input.get('ny', None)
    nx = input.get('nx', None)
    data = input.get('data', None)
    result = output.get("result", None)
    locations = oe.get("locations", None)
    max_error = oe.get("max_error", None)
    max_error_limit = oe.get('max_error_limit', None)
    gvfa_error = oe.get("gvfa_error", None)
    gvfa_error_limit = oe.get('gvfa_error_limit', None)

    if ny is not None and nx is not None:
        with builder.paragraph() as txt:
            txt += domain.gettext('function_called_with_params')

        with builder.list(style="compact") as lst:
            lst.add_item(f'ny = {ny}')
            lst.add_item(f'nx = {nx}')

        if data is not None:
            with builder.text() as txt:
                txt += domain.gettext('input_data_like')

            with builder.matrix(ny, nx) as mat:
                for i in range(ny):
                    for j in range(nx):
                        mat.entry(i, j, safeprint(safeget(data, i, j)))

        if result is not None:
            with builder.text() as txt:
                txt += domain.gettext('output')

            with builder.matrix(ny, ny) as mat:
                for i in range(ny):
                    for j in range(ny):
                        v = safeget(result, i, j)
                        if j < i:
                            mat.entry(i, j, safeprint(v), style="dim")
                        elif oe.get('locations') and saferatio(
                                safeget(locations, i, j), max_error_limit, 1):
                            mat.entry(i, j, safeprint(v), style="correct")
                        elif oe.get('locations') and saferatio(
                                safeget(locations, i, j), max_error_limit,
                                100):
                            mat.entry(i,
                                      j,
                                      safeprint(v),
                                      style="slightlywrong")
                        else:
                            mat.entry(i, j, safeprint(v), style="verywrong")

            if max_error_limit and locations:
                with builder.paragraph() as txt:
                    txt += domain.gettext('cells_wrong_intro')
                    txt += StringNode(domain.gettext('blue_shading'),
                                      style="elementexample slightlywrong")
                    txt += domain.gettext('cells_wrong_slightlywrong')
                    txt += StringNode(domain.gettext('orange_shading'),
                                      style="elementexample verywrong")
                    txt += domain.gettext('cells_wrong_verywrong')

    if safenum(max_error) > 0 and safenum(max_error_limit) > 0:
        with builder.paragraph() as txt:
            txt += domain.gettext('largest_error_intro')
            txt += StringNode(safereadable(max_error), style="strong")
            txt += domain.gettext(
                'largest_error_limit',
                max_error_limit=safereadable(max_error_limit))
            txt += domain.gettext(
                'largest_error_ratio',
                error_ratio=safereadable(max_error / max_error_limit),
            )

        if saferatio(max_error, max_error_limit, 100):
            with builder.paragraph() as txt:
                txt += domain.gettext('errors_maybe_rounding')

    elif safenum(gvfa_error) and safenum(gvfa_error_limit):
        with builder.paragraph() as txt:
            txt += domain.gettext(
                'probabilistic_error_ratio',
                error_ratio=safereadable(gvfa_error / gvfa_error_limit),
            )

    return builder.build()
