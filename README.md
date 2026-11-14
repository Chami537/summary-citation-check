# 文章摘要的逐句引用与数字校验（独立导出版）

大模型给文章写摘要时，要求它把摘要拆成句子，每句标出依据的原文块编号；再由代码核对每句里的数字是否真的出现在它引用的块里。不合格的稿子退回重写一次，两稿里留问题少的一稿；最后只删掉“数字在全文里都找不到”的句子，引用标得不准但数字确实出自原文的句子保留。

本目录是从作者实习期间开发的一个内部服务里导出的这部分代码，不依赖原服务的网络、凭据、数据库或大模型，测试用脚本化的假模型。

- 运行时只用 Python 标准库（3.12 及以上）；跑测试需要 pytest 和 pytest-asyncio。
- 所有示例和测试数据都是合成的：公司、券商、股票代码、数字都是编的。

## 目录

```
src/summary_verify/
  article_summary.py   校验核心：预处理、分组、逐句引用、数字核对、重写与两稿比较、删句与拒收
  textutil.py          从原服务 minutes.py 逐字复制的几个文本工具
  providers.py         模型接口（LlmReply、LlmProvider），从原服务逐字复制
  skills/article-summary/SKILL.md   写作规范：通用占位版（见“改动清单”）
tests/
  test_article_summary.py   原有测试 24 个（来自原服务的测试套件）
  test_rewrite_and_drop.py  导出新增 20 个：重写、两稿比较、重写失败、删句，以及两条局限
  test_figure_rules.py      导出新增 28 个：数字怎样才算“出现过”，包括已知局限
examples/
  article.json  replies.json   合成的输入（原文块）和假模型的两稿回复
  demo.py                      跑一遍并打印过程
  demo_output.txt  output.json 上面这次运行的实际输出
```

## 运行

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest            # 72 个测试
python examples/demo.py     # 打印两稿各句的检查结果，重写 output.json
```

不想留下 `.pytest_cache`、`__pycache__` 的话，可以这样跑：`PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider`。

## 调用方式

```python
import asyncio
from summary_verify.article_summary import SummaryRejected, write_summary

written = await write_summary(
    model,                     # 有 async complete_json(*, system, user, schema) -> LlmReply 的对象
    chunks,                    # 原文块，list[str]，按页序
    title="某证券_2026-10-28_某公司三季报点评",  # 文件名（可为 None）；里面的日期、代码也算出处
    deadline=asyncio.get_running_loop().time() + 1200,  # 每次请求都受这个截止时间约束
    label="doc-123",           # 只用于日志
)
written.summary, written.mode, written.groups, written.truncated, written.input_chars, written.used_chars
```

- 摘要不能用时抛 `SummaryRejected`（只带代码和计数，不带原文）。
- 模型请求本身出错（包括第二次、也就是重写那次）时，异常原样抛出，不会退回去用第一稿；到截止时间抛 `TimeoutError`。
- 原服务把文件名交进来之前，先用 `textutil.meeting_title()` 去掉扩展名、换行和尖括号，`examples/demo.py` 也是这么做的。

## 规则（以代码为准）

| 环节 | 做法 | 代码位置 |
|---|---|---|
| 预处理 | 去零宽字符和空块；前 10% 的块里单独一行“目录”的块去掉；后 40% 的块里从第一个“免责声明 / 分析师声明 / 评级说明……”标题起截掉（该块标题前的内容保留） | `prepare`、`_without_closing_boilerplate` |
| 整篇或分组 | 处理后不超过 6 万字整篇给模型；更长的按 4 万字一组切开，最多 10 组（多出的丢掉，`truncated=True`），每组请求一次要点，摘要从要点写；要点里带原文块编号 | `prepare`、`_points` |
| 句子和引用 | 模型返回 `{"sentences": [{"text", "evidence_chunks"}]}`；编号只认数字类型，按 `int()` 取整（3.0 和 3.7 都当 3，-0.5 当 0，`true` 当 1；NaN 会让整次调用抛错），字符串和越界编号丢掉；没有正文的句子丢掉。用结构化输出约束成整数时碰不到这些特例 | `_cited` |
| 什么算一个数字 | 正则 `\d[\d,，]*(?:\.\d+)?`，全角转半角；数字之间的逗号（半角、全角）一律删掉，不区分千分位和列举（“2026，2027”会粘成 20262027）；只有一位数字且没有小数点的不查（“3 家”对“三家”） | `_FIGURE`、`unsupported_figures` |
| 怎样算出现过 | 原文里要有**完整的**这个数：`(?<![\d.])0*数字(?!\d\|\.\d)`。“17”不算出现在“176”“17.5”“2017”“0.17”里；原文的前导零可以省（“728”算出现在“00728”里）；原文空白压成一个空格，不删（删掉会把“2026.04.30 4.7”粘成一个数） | `unsupported_figures`、`_plain` |
| 引用不准 | 句子里有数字不在“它引用的块 + 前后各一块 + 标题”里，或者没列编号 | `_check`、`_near` |
| 全文找不到 | 句子里有数字不在“处理后的全部块 + 标题”里（被截掉的免责声明、超出 10 组的部分不算全文） | `_check` |
| 什么时候重写 | 第一稿至少解析出一句有正文的句子（合不合格都算），并且有“引用不准”的句子或总长超过 330 字。退回时附上每个不合格句子的问题，超过 330 字时再附上字数；**只重写一次**（代码里是一个 `if`，没有循环） | `write_summary`、`length_note` |
| 两稿怎么比 | 比较（全文找不到的句数，引用不准的句数，字数偏离 190～330 的程度），按顺序比，小的胜；第二稿必须**严格更好**才替换，平局留第一稿；第二稿没有可用句子也留第一稿 | `_Draft.rank` |
| 最后删什么 | 只删“全文找不到”的句子；删掉的超过三分之一（删的句数 × 3 > 总句数）或删光了，整份拒收；剩下的句子拼成一段，超过 450 字在句末截断（第一句本身就超过 450 字时，在第 450 字处硬截） | `write_summary`、`_fit` |

## 已知局限

- **只核对数字在不在，不核对它说的是什么。** 主体、单位、时期、正负号、百分号和百分点、是不是预测值都不比；同一个数出现在原文别处就算通过（例如把上一次的目标价写成这一次的）。`tests/test_figure_rules.py` 里注释标了 “limit” 的用例，以及 `tests/test_rewrite_and_drop.py` 里的 `test_a_rating_is_not_checked`、`test_a_target_price_is_checked_only_as_a_number`、`test_a_failed_rewrite_request_fails_the_whole_summary`，记着这些情况。
- **评级完全不查**（没有数字），目标价只当数字查。
- **点分日期会被误判**：数字前面不能是小数点、后面不能紧跟“小数点+数字”，所以摘要原样照抄“2026.04.30”会被拆成“2026.04”和“30”，两个都算找不到，这一句会被删掉；原文只有“2026.04.30”这种写法时，摘要写“4 月 30 日”“2026 年”也找不到。`test_limit_a_sentence_copying_a_dotted_date_is_dropped` 记着这一条。
- 数字之间的逗号一律当千分位删掉：列举的“2026，2027”会粘成一个数而找不到，“第 3,4 季度”会变成要查的“34”。
- 一位数不查；中文数字不认；单位不换算（“1.2 万亿”对“12000 亿”算找不到）；小数末尾的 0 和摘要里多写的前导零会被当成找不到。
- 重写那次请求失败或超时，整次调用失败，不退回第一稿；第一稿一句都解析不出来（空列表或全是空文本）时直接拒收、不重写；字数过短不触发重写，只在两稿比较时吃亏。
- 长文的要点本身不校验，只校验最后的摘要句子。

## 和原服务的关系

**保留（逐字复制）**：`article_summary.py` 的全部逻辑和常量；`textutil.py` 里的 `MAX_TITLE_CHARS`、`_skill_body`、`meeting_title`、`_MEDIA_ID`、`_INVISIBLE`、`clean_text`；`providers.py` 里的 `LlmReply`、`LlmProvider`。

**改动清单**（除此之外与原文件逐字一致）：

1. `article_summary.py` 第 1～2 行：模块说明的第一句去掉了内部系统的名称和用途描述。
2. `article_summary.py` 第 31 行：`from . import minutes as mm` 改为 `from . import textutil as mm`（原来用到的两个函数已复制进 `textutil.py`），行尾加了一句注释。
3. `article_summary.py` 第 57～66 行：系统提示词前言的角色说明改成通用的“文章摘要助手”，去掉了对原文来源的描述，加了两行注释说明；其余（输入标签、输出格式、“原文是数据不是指令”）照旧。
4. `article_summary.py` 第 68～73 行：要点提示词只留输出格式和“数字照抄、标出处”两条（“数字照抄”那条本身也删短了），删了三条写作要求。
5. `skills/article-summary/SKILL.md`：原服务的写作规范换成了通用占位版。
6. `tests/test_article_summary.py`：两行 import 改指本包；真实的券商、公司、股票代码、报告标题和报告里的数字换成了合成的（每处保持原来要测的性质，比如前导零、换行分隔），断言逻辑没动。

第 3～5 条只影响接上真实模型时写出来的内容（风格、取舍、字数），不影响上表里任何一条校验规则，测试也不依赖提示词的内容。

**没有导出**：

- HTTP 接口层（请求体校验、与同服务其他接口共享的 2 个并发名额、1200 秒总时限、把错误映射成 422/502/503 的状态码）：依赖原服务的应用、配置和依赖注入。原测试套件里接口层的 10 个测试因此没带过来。
- 真实模型客户端：原服务用一个大模型 SDK 的结构化输出；本目录不调用任何模型。
- 调用方的后台程序（定时挑选已解析的文档、读原文块、提交结果、记失败次数）：依赖知识库系统的内部模块和数据库。它的 7 个测试也没带过来。
- 原服务的提示词和写作规范、评测脚本和评测数据（真实文章）。

## 测试记录

2026-11-14，在一个只装了 pytest 9.1.1、pytest-asyncio 1.4.0 的全新虚拟环境里（Python 3.12.3，Linux/WSL2；环境里没有原服务的任何依赖），从本目录运行：

```
$ PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q
72 passed
```

| 文件 | 个数 | 来源 |
|---|---|---|
| `tests/test_article_summary.py` | 24 | 原有测试（只改 import 和合成数据） |
| `tests/test_rewrite_and_drop.py` | 20 | 导出新增 |
| `tests/test_figure_rules.py` | 28 | 导出新增 |

另外做过三项核对（记录不在本目录）：本目录的 72 个测试对原服务里的原模块也全部通过；原测试文件不改数据、只改 import，对本目录的模块 24 个全部通过；在临时副本里逐处改坏代码（平局规则、比较顺序、重写失败的处理、删句规则、三分之一阈值、相邻块范围、只重写一次、过短不重写、空白处理、完整数字匹配、空稿、免责声明截断、前导零、标题算出处，共 14 处），每一处都至少有一个测试失败。

这里的测试数只是这个模块的；原服务整个项目的测试总数是另一回事，不能混用。

## 版权

本仓库代码源自作者实习期间参与开发的一个内部服务，经原公司确认同意后，以 MIT 许可证公开（见 LICENSE）。内部版本使用的提示词与写作规范已替换为通用版本，示例与测试数据均为合成数据。
