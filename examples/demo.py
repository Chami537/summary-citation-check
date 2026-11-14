"""Run the exported check once on made-up data, with a scripted fake model instead of a real one.

    python examples/demo.py            # from the project root; prints a trace, writes examples/output.json

The model's replies come from replies.json in order. The per-draft verdicts below are worked out with the
module's own _cited/_check, only to show them; the summary itself is what write_summary returns.
"""

import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from summary_verify import article_summary as asum  # noqa: E402
from summary_verify.providers import LlmReply  # noqa: E402
from summary_verify.textutil import meeting_title  # noqa: E402


class ScriptedModel:
    name, model, region = "scripted", "scripted-fake", ""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    async def complete_json(self, *, system, user, schema):
        self.requests.append(user)
        return LlmReply(self.replies.pop(0))


def verdicts(reply, prepared, title):
    draft = asum._check(asum._cited(reply["sentences"], len(prepared.chunks)), prepared.chunks, title)
    rows = []
    for k, (text, refs) in enumerate(draft.sentences):
        if k in draft.unfounded:
            verdict = "全文找不到"
        elif k in draft.failing:
            verdict = "引用不准"
        else:
            verdict = "通过"
        rows.append({"sentence": text, "evidence_chunks": refs, "verdict": verdict})
    return {"rank": list(draft.rank()), "sentences": rows}


async def main():
    article = json.loads((HERE / "article.json").read_text(encoding="utf-8"))
    drafts = json.loads((HERE / "replies.json").read_text(encoding="utf-8"))["drafts"]
    title = meeting_title(article["file_name"])  # the extension goes, as the service's endpoint did
    model = ScriptedModel(drafts)
    loop = asyncio.get_running_loop()
    written = await asum.write_summary(model, article["chunks"], title=title, deadline=loop.time() + 30, label="demo")

    prepared = asum.prepare(article["chunks"])
    checked = [verdicts(d, prepared, title) for d in drafts[: len(model.requests)]]
    feedback = model.requests[1].split("\n\n", 1)[1] if len(model.requests) > 1 else None
    result = {
        "title": title,
        "chunks_in": len(article["chunks"]),
        "chunks_used": len(prepared.chunks),
        "model_requests": len(model.requests),
        "rewrite_request_ends_with": feedback,
        "drafts": checked,
        "kept_draft": None,
        "summary": written.summary,
        "mode": written.mode,
        "groups": written.groups,
        "truncated": written.truncated,
        "input_chars": written.input_chars,
        "used_chars": written.used_chars,
    }
    # which draft write_summary kept: the one whose sentences, less the unfounded, make the summary
    for n, d in enumerate(checked, 1):
        if "".join(s["sentence"] for s in d["sentences"] if s["verdict"] != "全文找不到") == written.summary:
            result["kept_draft"] = n

    print(f"标题：{title}")
    print(f"原文 {result['chunks_in']} 块，去掉文末免责声明后用 {result['chunks_used']} 块；向模型请求 {result['model_requests']} 次")
    for n, d in enumerate(checked, 1):
        print(f"\n第{n}稿（排序键 全文找不到/引用不准/字数偏离 = {tuple(d['rank'])}）")
        for k, s in enumerate(d["sentences"], 1):
            print(f"  {k}. [{s['verdict']}] {s['sentence']}  引用 {s['evidence_chunks']}")
    if feedback:
        print(f"\n退回重写时附上的话：{feedback}")
    print(f"\n保留第{result['kept_draft']}稿；其中“全文找不到”的句子删去，“引用不准”的句子保留：\n{written.summary}")
    (HERE / "output.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
