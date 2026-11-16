import math

import pandas as pd

from nflsim import ui


def test_numbers_follow_the_style_guide():
    assert ui.signed(3.46) == "+3.5"
    assert ui.signed(-3.46) == "−3.5"
    assert ui.signed(0.04) == "0.0"
    assert ui.number(27.84) == "27.8"
    assert ui.percent(0.6728) == "67.3%"
    assert ui.signed_percent(-0.012) == "−1.2%"
    assert ui.signed_percent(0.0) == "0.0%"


def test_missing_values_show_a_dash():
    for text_format in (ui.signed, ui.number, ui.percent, ui.signed_percent):
        assert text_format(math.nan) == ui.BLANK


def test_shading_runs_from_orange_through_gray_to_blue():
    assert "rgb(241, 242, 244)" in ui.shade(0)
    assert "rgb(31, 95, 174)" in ui.shade(2) and "color:#fff" in ui.shade(2)
    assert "rgb(200, 89, 26)" in ui.shade(-1)
    assert ui.shade(math.nan) == ""


def test_shaded_text_is_the_more_readable_of_white_and_ink():
    assert ui.INK in ui.shade(0.2)
    assert "#fff" in ui.shade(1)


def test_styled_frame_formats_and_shades_only_the_named_columns():
    frame = pd.DataFrame({"team": ["TB", "KC"], "spread": [9.0, math.nan], "win": [0.5, 0.6]})
    html = ui.styled_frame(frame, {"spread": ui.signed, "win": ui.percent}, {"spread": (0.0, ui.SPREAD_SCALE)}).to_html()
    assert "+9.0" in html and "50.0%" in html and ui.BLANK in html
    assert html.count("background") == 1
