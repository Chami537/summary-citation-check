"""Article summaries: what the model gets from the chunks, and the checks on its cited sentences that keep
made-up and misplaced figures out."""

import asyncio

import pytest
from summary_verify import article_summary as asum
from summary_verify.providers import LlmReply

FOOTER = "请务必阅读正文之后的免责声明及其项下所有内容"


def body(n, size=100):
    return [f"第{i}块" + "正文" * size for i in range(n)]


def text(prepared):
    return "\n".join(prepared.chunks)


def test_short_article_goes_whole_without_toc_and_closing_boilerplate():
    chunks = ["目 录\n一、行业概况 ...... 3", f"投资要点：收入增长 25.3%。\n{FOOTER}", *body(8)]
    chunks += ["分析师声明\n本人具有中国证券业协会授予的证券投资咨询执业资格", "评级说明\n买入：……"]
    prepared = asum.prepare(chunks)
    assert prepared.groups == [range(9)] and not prepared.truncated
    assert (
        "目 录" not in text(prepared)
        and "分析师声明" not in text(prepared)
        and "评级说明" not in text(prepared)
    )
    assert FOOTER in text(prepared)  # a footer on the first page is not where the disclaimers start
    assert prepared.input_chars == prepared.used_chars == sum(len(c) for c in chunks[1:-2])


def test_what_precedes_the_disclaimers_in_their_chunk_is_kept():
    chunks = [
        *body(8),
        "5、风险提示\n原材料波动风险。\n特别声明\n《证券期货投资者适当性管理办法》……",
        "免责声明……",
    ]
    prepared = asum.prepare(chunks)
    assert prepared.chunks[-1] == "5、风险提示\n原材料波动风险。"


def test_boilerplate_headings_early_in_the_article_are_kept():
    chunks = ["一、免责声明的法律效力\n本文讨论免责声明条款", *body(9)]
    assert "免责声明的法律效力" in text(asum.prepare(chunks))


def test_long_article_is_cut_into_groups_in_page_order(monkeypatch):
    monkeypatch.setattr(asum, "MAX_WHOLE_CHARS", 1000)
    monkeypatch.setattr(asum, "GROUP_CHARS", 500)
    prepared = asum.prepare(body(10, size=100))  # 10 chunks of about 200 characters
    assert prepared.groups == [range(0, 2), range(2, 4), range(4, 6), range(6, 8), range(8, 10)]


def test_groups_past_the_limit_are_left_out(monkeypatch):
    monkeypatch.setattr(asum, "MAX_WHOLE_CHARS", 1000)
    monkeypatch.setattr(asum, "GROUP_CHARS", 500)
    monkeypatch.setattr(asum, "MAX_GROUPS", 2)
    prepared = asum.prepare(body(10, size=100))
    assert prepared.truncated and len(prepared.groups) == 2 and len(prepared.chunks) == 4
    assert prepared.input_chars > prepared.used_chars


@pytest.mark.parametrize(
    "summary, missing",
    [
        (
            "收入 1,234.5 亿元，同比增长 25.3%，目标价 ８８ 元。",
            [],
        ),  # thousands separators, full-width digits
        ("公司有 3 家工厂。", []),  # single digits are not checked: the article may say 三家
        ("预计 2027 年净利润 12.6 亿元。", ["2027", "12.6"]),
        ("二手房成交同比增长 17%，均价 25.3 元。", ["17"]),  # whole numbers only: not in 176 or 17.5
        ("毛利率 25 %。", ["25"]),  # nor 25 in 25.3
        ("示例公司（代码 512）。", []),  # a stock code without its leading zeros
        ("净利润3.1亿元、3.6亿元。", []),  # a line break between figures does not glue them
    ],
)
def test_figures_must_occur_as_whole_numbers(summary, missing):
    source = "2025年收入1234.5亿元，同比增长25.3 %。目标价88元，三家工厂。同比增长176%，单价17.5元。00512 3.1亿元、\n2025.12.31 \n3.6亿元"
    assert asum.unsupported_figures(summary, source) == missing


class Llm:
    name, model, region = "fake", "fake-model", ""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    async def complete_json(self, *, system, user, schema):
        self.calls.append((system, user, schema))
        return LlmReply(self.replies.pop(0))


def write(llm, chunks, title="示例证券_2026-10-13_示例乳业(000000)公司点评"):
    deadline = asyncio.get_running_loop().time() + 60
    return asum.write_summary(llm, chunks, title=title, deadline=deadline, label="test")


ARTICLE = [
    "示例乳业(000000)2025年收入120.4亿元，同比增长4.1%。",
    "维持买入评级，目标价16.5元（前值19.2元）。",
    *body(3),
]


def says(*sentences):
    """A model reply: each sentence with the chunk numbers it cites."""
    return {"sentences": [{"text": t, "evidence_chunks": refs} for t, refs in sentences]}


GOOD_SENTENCES = [
    ("示例乳业2025年收入120.4亿元，同比增长4.1%。", [0]),
    ("报告维持买入评级，目标价下调至16.5元。", [1]),
]
GOOD = says(*GOOD_SENTENCES)


async def test_whole_article_one_request_with_numbered_chunks():
    llm = Llm(GOOD)
    written = await write(llm, ARTICLE)
    assert (
        written.summary == "示例乳业2025年收入120.4亿元，同比增长4.1%。报告维持买入评级，目标价下调至16.5元。"
    )
    assert (written.mode, written.groups, written.truncated) == ("whole", 1, False)
    [(system, user, schema)] = llm.calls
    assert system == asum.SYSTEM_PROMPT and schema == asum.SUMMARY_SCHEMA
    assert user.startswith("<title>示例证券_2026-10-13_示例乳业(000000)公司点评</title>\n<article>\n[0] 示例乳业")
    assert "\n[1] 维持买入评级" in user


async def test_a_figure_from_another_chunk_is_sent_back_and_the_better_draft_kept():
    misplaced = says(("示例乳业收入120.4亿元，目标价16.5元。", [3]), ("维持买入评级。", [1]))
    llm = Llm(misplaced, GOOD)
    written = await write(llm, ARTICLE)
    assert written.summary.startswith("示例乳业2025年收入120.4亿元")
    assert "第1句的数字 120.4、16.5 在它引用的 [3] 里找不到" in llm.calls[1][1]


async def test_figures_from_the_next_chunk_or_the_title_hold():
    cover = says(("示例证券2026年10月13日覆盖示例乳业，2025年收入120.4亿元，目标价16.5元。", [0]))
    written = await write(Llm(cover), ARTICLE)  # 16.5 is in chunk 1, the date in the title
    assert written.summary.startswith("示例证券2026年10月13日")


async def test_a_figure_two_chunks_away_does_not_hold():
    far = says(("示例乳业收入120.4亿元。", [2]), ("报告维持买入评级。", [1]), ("报告看好低温奶。", [3]))
    llm = Llm(far, GOOD)
    await write(llm, ARTICLE)
    assert "第1句的数字 120.4 在它引用的 [2] 里找不到" in llm.calls[1][1]


async def test_a_sentence_without_citations_is_sent_back():
    llm = Llm(says(("示例乳业收入120.4亿元。", []), ("维持买入评级。", [1]), ("报告看好低温奶。", [2])), GOOD)
    await write(llm, ARTICLE)
    assert "第1句没有列出原文编号" in llm.calls[1][1]


async def test_a_failing_sentence_left_after_asking_is_dropped():
    one_bad = says(
        ("示例乳业2025年收入120.4亿元。", [0]),
        ("预计2027年收入200亿元。", [0]),
        ("报告维持买入评级。", [1]),
        ("目标价16.5元。", [1]),
    )
    written = await write(Llm(one_bad, one_bad), ARTICLE)
    assert written.summary == "示例乳业2025年收入120.4亿元。报告维持买入评级。目标价16.5元。"


async def test_a_miscited_sentence_left_after_asking_stays():
    miscited = says(("示例乳业2025年收入120.4亿元。", [0]), ("报告维持买入评级，目标价16.5元。", [3]))
    written = await write(Llm(miscited, miscited), ARTICLE)  # 16.5 is in chunk 1, not near 3
    assert written.summary == "示例乳业2025年收入120.4亿元。报告维持买入评级，目标价16.5元。"


async def test_too_many_failing_sentences_are_rejected():
    bad = says(("预计2027年收入200亿元。", [0]), ("报告维持买入评级。", [1]))
    with pytest.raises(asum.SummaryRejected, match="1 of 2 sentences do not hold"):
        await write(Llm(bad, bad), ARTICLE)


async def test_a_long_draft_is_sent_back_and_a_short_rewrite_loses():
    long = says(("示例乳业2025年收入120.4亿元。", [0]), *[("公司持续推进低温奶战略。", [2])] * 30)
    short = says(("示例乳业收入120.4亿元。", [0]))
    llm = Llm(long, short)
    written = await write(llm, ARTICLE)
    assert "太长了：请保留核心结论和主要理由" in llm.calls[1][1]
    # the rewrite is far too short: the long draft is kept, cut after a whole sentence
    assert written.summary.startswith("示例乳业2025年收入120.4亿元。公司持续")
    assert len(written.summary) <= asum.SUMMARY_MAX_CHARS and written.summary.endswith("。")


async def test_a_long_draft_rewritten_well_is_replaced():
    long = says(("示例乳业2025年收入120.4亿元。", [0]), *[("公司持续推进低温奶战略。", [2])] * 30)
    fits = says(*GOOD_SENTENCES, *[("公司持续推进低温奶战略，产品结构不断优化。", [2])] * 8)  # about 220
    written = await write(Llm(long, fits), ARTICLE)
    assert (
        written.summary.startswith("示例乳业2025年收入120.4亿元，同比增长4.1%。") and len(written.summary) < 330
    )


async def test_long_article_points_per_group_with_their_chunk_numbers(monkeypatch):
    monkeypatch.setattr(asum, "MAX_WHOLE_CHARS", 1000)
    monkeypatch.setattr(asum, "GROUP_CHARS", 500)
    llm = Llm(
        *({"points": [{"text": f"第{k}部分要点", "evidence_chunks": [2 * k]}] * 9} for k in range(5)),
        says(("全文摘要。", [0, 2])),
    )
    written = await write(llm, body(10, size=100))
    assert (written.mode, written.groups) == ("grouped", 5)
    assert [c[2] for c in llm.calls] == [asum.POINTS_SCHEMA] * 5 + [asum.SUMMARY_SCHEMA]
    assert '<article part="2/5">\n[2] 第2块' in llm.calls[1][1]
    final = llm.calls[-1][1]
    assert final.count("[第1部分]") == 8 and "第1部分要点（原文编号：2）" in final and "<article" not in final


async def test_nothing_but_boilerplate_is_rejected_before_asking():
    llm = Llm()
    with pytest.raises(asum.SummaryRejected, match="no text"):
        await write(llm, ["  ", "​"])
    assert llm.calls == []
