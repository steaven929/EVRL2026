"""Console configuration shared by executable entry points."""


def configure_utf8_console(stream):
    """Use UTF-8 and flush each write so IDE consoles update immediately."""
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return stream
    try:
        reconfigure(
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
            write_through=True,
        )
    except (AttributeError, ValueError):
        pass
    return stream
