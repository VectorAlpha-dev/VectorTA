import sys


def pytest_configure() -> None:
    """
    Compatibility shim: historical tests import `my_project`, but the published
    Python extension module is `vector_ta`.
    """
    import vector_ta

    sys.modules.setdefault("my_project", vector_ta)

