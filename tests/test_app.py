import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest


@pytest.mark.acceptance
def test_app_loads_without_exceptions() -> None:
    at = AppTest.from_file("app.py").run()
    assert len(at.exception) == 0
