from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "ui/app.py"


def select(app, label, value):
    next(widget for widget in app.selectbox if widget.label == label).set_value(value).run(timeout=20)
    assert not app.exception


def test_manual_app_initial_state():
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "Base Siege"
    assert app.session_state["game"].state.spent == 0
    assert app.session_state["game"].state.tick == 0


def test_heuristic_replay_is_labelled_and_complete():
    app = AppTest.from_file(str(APP)).run(timeout=20)
    select(app, "Mode", "Policy replay")
    # Saved models change the UI default; this test exercises the baseline.
    select(app, "Policy", "Heuristic")
    next(button for button in app.button if button.label == "Generate replay").click().run(timeout=20)
    assert not app.exception
    assert not app.error, [error.value for error in app.error]
    replay = app.session_state["replay"]
    assert replay["policy"]["name"] == "Heuristic"
    assert replay["summary"]["finish_reason"]
    assert replay["summary"]["spent"] == 199


def test_evidence_modes_handle_present_or_absent_artifacts():
    app = AppTest.from_file(str(APP)).run(timeout=20)
    mode = next(widget for widget in app.selectbox if widget.label == "Mode")
    assert "Experiment results" not in mode.options
    select(app, "Mode", "Saved replay")
    select(app, "Mode", "Manual play")
