"""The line-chart report: charts are drawn into the page, readable with no script and no
stylesheet, and the end labels never overlap."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from hiring_bias_mitigation.viz import line_report as L

SVG = "{http://www.w3.org/2000/svg}"


def _spec(**overrides) -> dict:
    spec = {
        "id": "acc", "title": "正确率", "subtitle": "说明", "x_label": "步",
        "x_ticks": [0, 50, 100], "x_min": 0, "x_max": 100,
        "y_min": 0.5, "y_max": 1.0, "y_ticks": [0.5, 0.75, 1.0], "percent": True,
        "series": [
            {"name": "录（期望）", "label": "录", "role": "a", "weight": "primary",
             "points": [[10, 0.90], [50, 0.95], [90, 0.971]]},
            {"name": "拒（期望）", "label": "拒", "role": "b", "weight": "primary",
             "points": [[10, 0.80], [50, float("nan")], [90, 0.965]]},
            {"name": "日志", "role": "neutral", "weight": "secondary",
             "points": [[x, 0.7 + x / 1000] for x in range(1, 101)]},
        ],
        "table": {"columns": ["步", "录"], "rows": [[10, "90.0%"]]},
    }
    spec.update(overrides)
    return spec


def _page(spec: dict | None = None) -> str:
    return L.render_report("标题", ["引言"], [spec or _spec()], [])


def _drawn(page: str, chart_id: str = "acc") -> ET.Element:
    """The chart as a viewer that runs no script shows it: the SVG already in the page."""
    without_scripts = re.sub(r"<script\b.*?</script>", "", page, flags=re.S)
    found = re.search(rf'<div class="plot" id="plot-{chart_id}">(<svg.*?</svg>)</div>',
                      without_scripts, re.S)
    assert found, "chart is not drawn into the page"
    return ET.fromstring(found.group(1))


def test_chart_is_drawn_without_scripts():
    svg = _drawn(_page())
    paths = svg.findall(f"{SVG}path")
    # one line per series, the gap in "拒" splitting its line into two runs
    assert len(paths) == 3
    assert [p.get("d").count("M") for p in paths] == [1, 1, 2]
    # markers on the primary series' finite points only
    assert len(svg.findall(f"{SVG}circle")) == 3 + 2
    text = "".join(svg.itertext())
    for label in ("50%", "75%", "100%", "0", "50", "100", "步"):
        assert label in text


def test_marks_carry_literal_colours():
    svg = _drawn(_page())
    for node in svg.iter():
        for attr in ("stroke", "fill"):
            value = node.get(attr)
            if value is not None and value != "none":
                assert re.fullmatch(r"#[0-9a-f]{6}", value), (node.tag, attr, value)
    strokes = {p.get("stroke") for p in svg.findall(f"{SVG}path")}
    assert strokes == {L.LIGHT["a"], L.LIGHT["b"], L.LIGHT["neutral"]}


def test_no_non_finite_coordinates():
    svg = _drawn(_page())
    for node in svg.iter():
        for value in node.attrib.values():
            assert not re.search(r"nan|inf", value, re.I), value


def test_stylesheet_themes_the_marks():
    page = _page()
    assert ".s-a { stroke: var(--a); }" in page
    assert L.DARK["a"] in page and "prefers-color-scheme: dark" in page


def test_end_labels_spread_apart_when_ends_collide():
    svg = _drawn(_page())
    ends = [t for t in svg.findall(f"{SVG}text") if t.get("class") == "c-end"]
    assert [t.text for t in ends] == ["录 97.1%", "拒 96.5%"]
    heights = sorted(float(t.get("y")) for t in ends)
    assert heights[1] - heights[0] >= L.LABEL_GAP - 1e-9


def test_dodge():
    assert L.dodge([50.0, 200.0], 14, 0, 300) == [50.0, 200.0]
    assert L.dodge([104.0, 100.0], 14, 0, 300) == pytest.approx([109.0, 95.0])
    three = L.dodge([100.0, 101.0, 102.0], 10, 0, 300)
    assert three == pytest.approx([91.0, 101.0, 111.0])
    edge = L.dodge([299.0, 298.0], 14, 0, 300)
    assert max(edge) <= 300 and abs(edge[0] - edge[1]) == pytest.approx(14)
    assert L.dodge([], 14, 0, 300) == []


def test_unlabelled_chart_keeps_the_narrow_margin():
    spec = _spec(series=[{"name": "日志", "role": "neutral", "weight": "secondary",
                          "points": [[0, 0.6], [100, 0.7]]}])
    assert L.geometry(spec)["r"] == L.MARGIN["r"]
    assert L.geometry(_spec())["r"] == L.LABEL_ROOM


def test_text_is_escaped_and_data_cannot_close_the_script():
    spec = _spec(title="a < b & c")
    spec["series"][0]["name"] = "</script><b>x"
    page = _page(spec)
    svg = _drawn(page)
    assert svg.get("aria-label") == "a < b & c"
    data = re.search(r'id="report-data">(.*?)</script>', page, re.S).group(1)
    assert "</script>" not in data and "<\\/script>" in data
