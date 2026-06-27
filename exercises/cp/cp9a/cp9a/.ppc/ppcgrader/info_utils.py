import math
from typing import Union, Any, List, Type


def readable(x):
    """
    Converts a number to string in non-scientific format in a way that ensures that small numbers are never rounded to
    zero, and large numbers are rounded to the nearest integer.
    """
    if abs(x) >= 100:
        return f'{x:.0f}'
    if abs(x) >= 10:
        return f'{x:.1f}'
    i = 0
    while x != 0 and float(f'{x:.{i}f}') == 0:
        i += 1
    i += 2
    return f'{x:.{i}f}'


def isnum(v):
    """
    Returns if `v` is of numeric type, i.e. an integer or a floating point number
    """
    return v is not None and (isinstance(v, int) or
                              (isinstance(v, float) and math.isfinite(v)))


def _multi_get(collection, indices):
    """
    Given a collection, and a sequence of indices `indices = (i1, ..., in)`, returns the subscript
    `collection[i1][...][in]`
    """
    for i in indices:
        collection = collection[i]
    return collection


def safeget(m, *indices):
    """
    Given a sequence of indices `i1, ..., ìn`, performs the lookup m[i1][...][in]. If this lookup is not
    possible, returns `None`
    """
    try:
        return _multi_get(m, indices)
    except (KeyError, IndexError, TypeError):
        return None


def safenum(v, default=0):
    """
    For numeric values, this is an identity operation. Any other value is replaced by default.
    """
    return v if isnum(v) else default


def safestr(v, default='–'):
    """
    Returns the default value of `v is None`, otherwise converts `v` to string.
    """
    return default if v is None else str(v)


def safeprint(v, fmt_float='{:+.8f}'):
    """
    If the given value is a floating point, it is returned using the corresponding format string.
    Otherwise, a string representation of the value is returned, with `None` replace by the string "-".
    """
    if isinstance(v, float):
        return fmt_float.format(v)
    elif isinstance(v, int):
        return '{:d}'.format(v)
    return safestr(v, default='–')


def saferatio(x, y, f):
    """
    Tests if the ratio `x / y` is smaller than `f`. If `x` or `y` is not a number, then the test always returns False.
    """
    if isnum(x) and isnum(y):
        # write as product to avoid division by zero.
        return x < f * y
    else:
        return False


def safereadable(v):
    """
    Converts the number into `readable` format, ensuring that `None` results in `-` and non-numeric
    inputs are converted to string, instead of causing an error.
    """
    if v is None:
        return '–'
    elif isnum(v):
        return readable(v)
    else:
        return str(v)


def render_explain_web(template: str, raw: dict, **kwargs):
    from jinja2 import Template
    from markupsafe import Markup

    return Markup(
        Template(template).render(
            input=raw.get("input", {}),
            output=raw.get("output", {}),
            oe=raw.get("output_errors", {}),
            **kwargs,
            safeget=safeget,
            safenum=safenum,
            safestr=safestr,
            saferatio=saferatio,
            safeprint=safeprint,
            safereadable=safereadable,
        ))


def get_system_language() -> str:
    """Determines the system language with a fallback to environment variables.

    Supports 'en' and 'fi'. Defaults to 'en' if detection fails or the 
    detected language is unsupported.

    Currently remains unused as grader is only in English at the moment.

    Returns:
        A two-letter language code (e.g., 'en', 'fi').
    """
    import os
    import locale

    try:
        # Attempts to get default locale, e.g., ('en_US', 'UTF-8')
        lang_code, _ = locale.getdefaultlocale()
        if lang_code:
            short_lang = lang_code.split('_')[0].lower()
            if short_lang in ['en', 'fi']:
                return short_lang
    except Exception:
        # Silently ignore locale errors to move to fallback
        pass

    # Fallback to standard Linux/Unix environment variables
    for var in ['LANGUAGE', 'LC_ALL', 'LC_MESSAGES', 'LANG']:
        val = os.environ.get(var)
        if val:
            short_lang = val.split('_')[0].lower()
            if short_lang in ['en', 'fi']:
                return short_lang
    # Default to
    return 'en'


def get_domain_class(mode: str) -> Type:
    """Factory that returns a Domain class based on the execution environment.
    
    IMPROVE THIS IF WORKING WITH GRADER TRANSLATIONS! 
    In 'web' mode, returns the Flask-Babel Domain. In other modes, returns a 
    custom shim that uses the standard library's gettext.

    Args:
        mode: The operation mode ('web' or 'cli'/other).

    Returns:
        A Domain class (not an instance) compatible with gettext calls.
    """
    if mode == "web":
        try:
            from flask_babel import Domain
            return Domain
        except ModuleNotFoundError:
            # In test or non-Flask environments, fall back to the local
            # gettext-based implementation so grader code still works.
            pass

    # Standard Library Implementation (CLI/Local Mode)
    import gettext
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Prefer compiled web translations during development (so grader
    # and exercise domains can be tested directly from the repo),
    # but fall back to the bundled ppcgrader/translations directory
    # when running from a packaged grader on student machines.
    repo_root = os.path.abspath(os.path.join(current_dir, os.pardir,
                                             os.pardir))
    web_build_translations = os.path.join(repo_root, "web", "ppcweb", "build",
                                          "translations")
    if os.path.isdir(web_build_translations):
        translations_path = web_build_translations
    else:
        translations_path = os.path.join(current_dir, 'translations')

    class LocalDomain:
        """Shim class to provide flask_babel.Domain functionality using gettext."""
        def __init__(
                self,
                translation_directories: Union[str,
                                               List[str]] = translations_path,
                domain: str = 'messages'):
            """Initializes the local translation domain.

            Args:
                translation_directories: Path(s) to search for .mo files.
                domain: The name of the translation domain (the .mo filename).
            """
            if isinstance(translation_directories, list):
                self.localedir = translation_directories[0]
            else:
                self.localedir = translation_directories
            self.domain = domain

        def gettext(self, message: str, **kwargs: Any) -> str:
            """Translates a message and interpolates variables.

            Args:
                message: The source string to translate.
                **kwargs: Variables for string formatting.

            Returns:
                The translated and formatted string.
            """
            # TODO: Fully translate grader to support Finnish
            lang = 'en'

            try:
                t = gettext.translation(self.domain,
                                        localedir=self.localedir,
                                        languages=[lang],
                                        fallback=True)
                translated = t.gettext(message)
            except Exception:
                translated = message

            if kwargs:
                try:
                    # Try printf-style formatting: "Hello %s" % "world"
                    return translated % kwargs
                except TypeError:
                    # Fallback to str.format: "Hello {name}".format(name="world")
                    return translated.format(**kwargs)
            return translated

    return LocalDomain


def get_domain(mode: str, domain: str = "grader") -> 'Domain':
    """This function returns the requested mode version of the requested domain as a Domain object.
    Defaults to the grader domain.

    Keyword arguments:
    mode -- mode of the requested object (web/term)
    domain -- string with the name of the domain

    Return: Domain object
    """

    DomainClass = get_domain_class(mode)
    return DomainClass(domain=domain)


def get_grader_domain(mode: str) -> 'Domain':
    return get_domain(mode)
