"""Added for the export (not in the service's test suite): the way write_summary sends a draft back, compares
the two drafts and drops sentences, pinned down call by call with a scripted fake model. Every case states
what the code does today, limits included (see the README's known limits). All data here is made up."""

import asyncio

import pytest
from summary_verify import article_summary as asum
from summary_verify.providers import LlmReply

TITLE = "示例证券_2026-10-13_示例乳业(000000)公司点评"
ARTICLE = [
    "示例乳业(000000)2025年收入120.4亿元，同比增长4.1%。",
    "维持买入评级，目标价16.5元（前值19.2元）。",
    "低温奶收入占比提升至35%，毛利率改善。",
    "风险提示：原奶价格波动。",
    "公司计划新建两座工厂。",
]


class Script:
    """A model that answers from a list: a dict is a reply, an exception is raised, a float is a delay (in
    seconds) before the next item is answered. Each request is recorded."""

    name, model, region = "fake", "fake-model", ""

    def __init__(self, *items):
        self.items = list(items)
        self.calls = []

    async def complete_json(self, *, system, user, schema):
        self.calls.append({"system": system, "user": user, "schema": schema})
        item = self.items.pop(0)
        if isinstance(item, float):
            await asyncio.sleep(item)
            item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return LlmReply(item)


def says(*sentences):
    return {"sentences": [{"text": t, "evidence_chunks": refs} for t, refs in sentences]}


def summarize(llm, chunks=ARTICLE, *, seconds=60.0):
    deadline = asyncio.get_running_loop().time() + seconds
    return asum.write_summary(llm, chunks, title=TITLE, deadline=deadline, label="test")


HOLDS = [
    ("示例乳业2025年收入120.4亿元，同比增长4.1%。", [0]),
    ("报告维持买入评级，目标价16.5元。", [1]),
    ("低温奶收入占比提升至35%。", [2]),
]
UNFOUNDED = ("预计2027年收入200亿元。", [0])  # neither figure is anywhere in the article or the title
MISCITED = ("目标价16.5元。", [4])  # 16.5 is in chunk 1, not in 3, 4 or the title


# --- when a draft goes back, and how often --------------------------------------------------------------


async def test_a_draft_that_holds_is_not_sent_back():
    llm = Script(says(*HOLDS))
    written = await summarize(llm)
    assert len(llm.calls) == 1 and written.summary.startswith("示例乳业2025年收入120.4亿元")


async def test_a_draft_is_sent_back_once_however_bad_the_second_one_is():
    bad = says(MISCITED, *HOLDS)
    llm = Script(bad, bad, says(*HOLDS))  # a third reply is there, but never asked for
    await summarize(llm)
    assert len(llm.calls) == 2 and len(llm.items) == 1


async def test_a_short_draft_alone_is_not_sent_back():
    llm = Script(says(("示例乳业收入120.4亿元。", [0])))  # 14 characters, below SHORT_CHARS
    written = await summarize(llm)
    assert len(llm.calls) == 1 and written.summary == "示例乳业收入120.4亿元。"


async def test_an_empty_first_draft_is_rejected_without_asking_again():
    llm = Script(says(), says(*HOLDS))
    with pytest.raises(asum.SummaryRejected, match="no sentence that holds"):
        await summarize(llm)
    assert len(llm.calls) == 1


async def test_the_request_to_rewrite_lists_each_problem_and_the_length():
    long_tail = [("公司持续推进低温奶战略。", [2])] * 26  # pushes the draft past LONG_CHARS
    first = says(("示例乳业收入120.4亿元。", []), ("报告维持买入评级，目标价18.0元。", [1]), *HOLDS, *long_tail)
    llm = Script(first, says(*HOLDS))
    await summarize(llm)
    second = llm.calls[1]["user"]
    assert second.startswith(llm.calls[0]["user"] + "\n\n上一版摘要的问题：")
    assert "第1句没有列出原文编号；第2句的数字 18.0 在它引用的 [1] 里找不到；一共 " in second
    assert "太长了" in second and second.endswith("请重写摘要。")
    assert llm.calls[1]["system"] == llm.calls[0]["system"] == asum.SYSTEM_PROMPT


# --- which of the two drafts is kept -------------------------------------------------------------------


async def test_the_draft_with_fewer_unfounded_sentences_wins_even_with_more_miscited_ones():
    first = says(UNFOUNDED, *HOLDS)  # 1 unfounded (and failing)
    second = says(MISCITED, ("报告维持买入评级。", [4]), ("低温奶收入占比提升至35%。", [0]), *HOLDS[:1])
    written = await summarize(Script(first, second))  # 0 unfounded, 2 miscited
    assert written.summary.startswith("目标价16.5元。报告维持买入评级。低温奶收入占比提升至35%。")


async def test_a_tie_keeps_the_first_draft():
    first = says(("示例乳业收入120.4亿元。", [3]))  # miscited, 14 characters
    second = says(("示例乳业营收120.4亿元。", [3]))  # the same problems and the same length
    written = await summarize(Script(first, second))
    assert written.summary == "示例乳业收入120.4亿元。"


async def test_a_rewrite_without_a_usable_sentence_keeps_the_first_draft():
    first = says(MISCITED, *HOLDS)
    malformed = {"sentences": [{"text": "   ", "evidence_chunks": [0]}, {"evidence_chunks": [1]}, "第三句"]}
    written = await summarize(Script(first, malformed))
    assert written.summary.startswith("目标价16.5元。")


# --- when the rewrite request itself fails ------------------------------------------------------------


async def test_a_failed_rewrite_request_fails_the_whole_summary():
    """The first draft is not used as a fallback: the second request's error is raised (the service's
    endpoint then answered 503 for a model-service failure, 502 for anything else)."""
    llm = Script(says(MISCITED, *HOLDS), ConnectionError("model service unreachable"))
    with pytest.raises(ConnectionError):
        await summarize(llm)
    assert len(llm.calls) == 2


async def test_a_rewrite_past_the_deadline_fails_the_whole_summary():
    llm = Script(says(MISCITED, *HOLDS), 0.5, says(*HOLDS))  # the second answer comes after the deadline
    with pytest.raises(TimeoutError):
        await summarize(llm, seconds=0.2)


# --- which sentences are dropped at the end -----------------------------------------------------------


async def test_unfounded_sentences_are_dropped_and_miscited_ones_kept():
    draft = says(UNFOUNDED, MISCITED, *HOLDS)  # 1 of 5 unfounded
    written = await summarize(Script(draft, draft))
    assert "200" not in written.summary and written.summary.startswith("目标价16.5元。")


async def test_one_unfounded_sentence_in_three_is_dropped_but_two_in_five_reject_the_summary():
    one_in_three = says(UNFOUNDED, *HOLDS[:2])
    written = await summarize(Script(one_in_three, one_in_three))
    assert written.summary == "示例乳业2025年收入120.4亿元，同比增长4.1%。报告维持买入评级，目标价16.5元。"
    two_in_five = says(UNFOUNDED, ("公司计划新建18座工厂。", [4]), *HOLDS)  # (a single digit is never checked)
    with pytest.raises(asum.SummaryRejected, match="2 of 5 sentences do not hold"):
        await summarize(Script(two_in_five, two_in_five))


async def test_every_figure_of_a_sentence_has_to_be_in_the_article():
    draft = says(("收入120.4亿元，净利润9.87亿元。", [0]), *HOLDS)  # 120.4 is there, 9.87 is not
    written = await summarize(Script(draft, draft))
    assert "9.87" not in written.summary and written.summary.startswith("示例乳业2025年")


async def test_a_figure_only_in_the_dropped_disclaimers_counts_as_not_in_the_article():
    chunks = [*ARTICLE, "行业保持增长。", "需求稳定。", "渠道调整完成。", "免责声明\n本报告有效期36个月。"]
    draft = says(("本报告有效期36个月。", [8]), *HOLDS)  # chunk 8 is cut off as boilerplate
    written = await summarize(Script(draft, draft), chunks)
    assert "36" not in written.summary


async def test_a_rating_is_not_checked():
    draft = says(("报告上调评级至增持。", [1]), *HOLDS)  # the article says 维持买入
    written = await summarize(Script(draft))
    assert written.summary.startswith("报告上调评级至增持。")


async def test_a_target_price_is_checked_only_as_a_number():
    draft = says(("报告给出目标价19.2元。", [1]), *HOLDS)  # 19.2 is the previous target, not the current one
    written = await summarize(Script(draft))
    assert written.summary.startswith("报告给出目标价19.2元。")


async def test_limit_a_sentence_copying_a_dotted_date_is_dropped():
    chunks = [*ARTICLE[:4], "截至2026.04.30，公司在手订单充足。"]
    draft = says(("截至2026.04.30，公司在手订单充足。", [4]), *HOLDS)  # copied word for word
    written = await summarize(Script(draft, draft), chunks)
    assert "在手订单" not in written.summary  # "2026.04" and "30" are not found as whole numbers


async def test_limit_a_first_sentence_past_the_cap_is_cut_mid_sentence():
    draft = says(("公司持续推进低温奶战略" * 50 + "。", [2]), *HOLDS)  # 551 characters before the others
    written = await summarize(Script(draft, draft))
    assert len(written.summary) == asum.SUMMARY_MAX_CHARS and not written.summary.endswith("。")


# --- a long article --------------------------------------------------------------------------------------


async def test_grouped_points_are_asked_once_per_group_and_reused_by_the_rewrite(monkeypatch):
    monkeypatch.setattr(asum, "MAX_WHOLE_CHARS", 800)
    monkeypatch.setattr(asum, "GROUP_CHARS", 500)
    chunks = [f"第{i}块收入{100 + i}.5亿元。" + "正文" * 100 for i in range(4)]  # two groups of two
    points = [{"points": [{"text": f"第{k}部分收入{100 + 2 * k}.5亿元", "evidence_chunks": [2 * k]}]} for k in range(2)]
    first = says(("收入100.5亿元。", [3]), ("收入102.5亿元。", [2]))  # the first sentence is miscited
    llm = Script(*points, first, says(("收入100.5亿元。", [0]), ("收入102.5亿元。", [2])))
    written = await summarize(llm, chunks)
    schemas = [c["schema"] for c in llm.calls]
    assert schemas == [asum.POINTS_SCHEMA, asum.POINTS_SCHEMA, asum.SUMMARY_SCHEMA, asum.SUMMARY_SCHEMA]
    assert llm.calls[3]["user"].startswith(llm.calls[2]["user"])  # the same points, not asked again
    assert (written.mode, written.groups, written.summary) == ("grouped", 2, "收入100.5亿元。收入102.5亿元。")


async def test_points_are_not_checked_but_a_summary_figure_taken_from_them_is(monkeypatch):
    monkeypatch.setattr(asum, "MAX_WHOLE_CHARS", 800)
    monkeypatch.setattr(asum, "GROUP_CHARS", 500)
    chunks = [f"第{i}块收入{100 + i}.5亿元。" + "正文" * 100 for i in range(4)]
    points = [{"points": [{"text": "收入999亿元", "evidence_chunks": [0]}]}, {"points": [{"text": "收入103.5亿元", "evidence_chunks": [3]}]}]
    draft = says(("收入999亿元。", [0]), ("第2块收入102.5亿元。", [2]), ("第3块收入103.5亿元。", [3]), ("第0块收入100.5亿元。", [0]))
    llm = Script(*points, draft, draft)
    written = await summarize(llm, chunks)
    assert "收入999亿元（原文编号：0）" in llm.calls[2]["user"]  # passed on as the model wrote it
    assert "999" not in written.summary  # but the sentence using it is dropped
