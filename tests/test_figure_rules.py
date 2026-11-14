"""Added for the export (not in the service's test suite): what unsupported_figures does and does not compare.
A figure is digits, with thousands separators and decimals; it has to occur in the source as a whole number.
Signs, units, percent signs, Chinese numerals and single digits are not compared: the cases marked "limit"
pass or fail today for that reason, and are here so the limits stay visible. All data here is made up."""

import pytest
from summary_verify.article_summary import unsupported_figures


@pytest.mark.parametrize(
    "text, source, missing",
    [
        # what it catches
        ("增长17%。", "增长176%", ["17"]),  # not part of a longer number
        ("增长17%。", "单价17.5元", ["17"]),  # nor the whole part of a decimal
        ("增长17%。", "2017年", ["17"]),  # nor the end of a year
        ("增长17%。", "增长0.17倍", ["17"]),  # nor the digits after a decimal point
        ("收入3.6亿元。", "2025.12.313.6亿元", ["3.6"]),  # glued to the figure before it
        ("收入1.2万亿元。", "收入12000亿元", ["1.2"]),  # no conversion between units
        ("预计12家。", "预计十二家", ["12"]),  # Chinese numerals are not read
        ("区间10-15%。", "区间10%", ["15"]),  # a range is two figures
        # what it lets through
        ("收入1,234.5亿元。", "收入1234.5亿元", []),  # thousands separators on either side
        ("收入1234亿元。", "收入1，234亿元", []),  # a full-width separator too
        ("目标价８８元。", "目标价88元", []),  # full-width digits
        ("2026年10月13日。", "2026-10-13", []),  # a date's parts, whatever separates them
        ("区间10-15%。", "区间10%至15%", []),
        ("代码512。", "代码00512。", []),  # leading zeros in the source may go
        ("增长17%。", "增长17", []),  # a figure at the end of a sentence
        # limits: compared as bare numbers
        ("同比下降-3.2%。", "同比增长3.2%", []),  # limit: the sign is not compared
        ("收入17亿元。", "收入17万元", []),  # limit: nor the unit
        ("增长17%。", "增长17个百分点", []),  # limit: nor percent against percentage points
        ("公司有5家工厂。", "公司有三家工厂", []),  # limit: single digits are never checked
        ("毛利率25%。", "2026年毛利率21%，目标价25元", []),  # limit: same number, different quantity
        # limits: written differently, so missing
        ("单价17.5元。", "单价17.50元", ["17.5"]),  # limit: a trailing zero makes another number
        ("单价17.50元。", "单价17.5元", ["17.50"]),
        ("代码00512。", "代码512", ["00512"]),  # limit: leading zeros may go only from the source
        # limits: a comma between figures is taken for a thousands separator
        ("预计2026，2027年。", "预计2026E、2027E", ["20262027"]),  # limit: two years glued into one figure
        ("第3,4季度。", "第三、四季度", ["34"]),  # limit: two single digits become one checked figure
        # limits: a dotted date
        ("截至2026.04.30。", "截至2026.04.30，", ["2026.04", "30"]),  # limit: missed even when copied
        ("4月30日，2026年。", "截至2026.04.30，", ["30", "2026"]),  # limit: nor are its parts found
    ],
)
def test_what_counts_as_the_same_figure(text, source, missing):
    assert unsupported_figures(text, source) == missing


def test_a_figure_is_reported_once_in_the_order_it_appears():
    assert unsupported_figures("收入88亿元、99亿元，再写一次88亿元。", "无") == ["88", "99"]
