"""Frozen catalog for the M020 novel-vertical SFT v7 data builder.

The catalog contains task wording and ontology labels only.  It deliberately
contains no free-standing factual answers: every answer produced by the
builder must be derived from an exact line in ``data/cloud_v4/train.txt``.
Prompt and answer-style banks are physically separated by split so their IDs
and wording fingerprints can be audited for leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


SCHEMA_VERSION: Final = "sft_v7_vertical/1.0"
MANIFEST_SCHEMA_VERSION: Final = "sft-v7-vertical-manifest/v1"
FROZEN_SEED: Final = 20260829

SPLITS: Final = ("train", "val", "public_diagnostic", "sealed_test")
SPLIT_RATIOS: Final = {
    "train": 0.80,
    "val": 0.08,
    "public_diagnostic": 0.06,
    "sealed_test": 0.06,
}

CORE = "parameter_core_fact_and_correction"
EVIDENCE = "single_passage_grounded_qa"
RAG = "multi_passage_rag_evidence_composition"
CHAT = "vertical_chat_multiturn_eos"
EXPRESSION = "novel_summary_rewrite_short_continuation"
BOUNDARY = "capability_boundary_clarification_evidence_request"

DIMENSION_TOTALS: Final = {
    CORE: 1800,
    EVIDENCE: 3200,
    RAG: 1400,
    CHAT: 1800,
    EXPRESSION: 1300,
    BOUNDARY: 500,
}


def _split_quota(total: int) -> dict[str, int]:
    return {
        "train": total * 80 // 100,
        "val": total * 8 // 100,
        "public_diagnostic": total * 6 // 100,
        "sealed_test": total * 6 // 100,
    }


DIMENSION_SPLIT_QUOTAS: Final = {
    dimension: _split_quota(total) for dimension, total in DIMENSION_TOTALS.items()
}
SPLIT_TOTALS: Final = {
    split: sum(DIMENSION_SPLIT_QUOTAS[dimension][split] for dimension in DIMENSION_TOTALS)
    for split in SPLITS
}

NEGATIVE_SHARE_NUMERATOR: Final = 7
NEGATIVE_SHARE_DENOMINATOR: Final = 40  # 17.5%
MINIMUM_MULTITURN_RECORDS: Final = 1200
MINIMUM_RAG_RECORDS: Final = 1000

BANNED_TEXT_MARKERS: Final = (
    "可以先",
    "先先",
    "审核占位",
    "等待审核",
    "现有正式训练语料",
    "Token",
    "Tokenizer",
    "BPE",
    "Embedding",
    "GPT原理",
    "监督微调",
    "训练Loss",
)

UNSAFE_SOURCE_MARKERS: Final = (
    "手机访问",
    "未完待续",
    "支持正版",
    "登陆com",
    "更新到",
    "推荐一本",
    "月票",
    "草你奶奶",
    "操你",
    "强奸",
    "性交",
    "乳房",
    "裸体",
    "一丝不挂",
)


@dataclass(frozen=True)
class CoreTerm:
    """A label used only to locate and balance corpus-backed examples."""

    label: str
    kind: str
    aliases: tuple[str, ...] = ()


CORE_TERMS: Final = (
    CoreTerm("萧炎", "人物"),
    CoreTerm("药老", "人物", ("药尘",)),
    CoreTerm("药尘", "人物", ("药老",)),
    CoreTerm("萧战", "人物"),
    CoreTerm("萧薰儿", "人物", ("薰儿",)),
    CoreTerm("纳兰嫣然", "人物"),
    CoreTerm("云韵", "人物"),
    CoreTerm("云山", "人物"),
    CoreTerm("美杜莎", "人物", ("彩鳞", "美杜莎女王")),
    CoreTerm("小医仙", "人物"),
    CoreTerm("海波东", "人物", ("冰皇",)),
    CoreTerm("韩枫", "人物"),
    CoreTerm("古河", "人物", ("丹王古河",)),
    CoreTerm("紫研", "人物", ("紫妍",)),
    CoreTerm("苏千", "人物"),
    CoreTerm("风尊者", "人物"),
    CoreTerm("古元", "人物"),
    CoreTerm("魂天帝", "人物"),
    CoreTerm("萧厉", "人物"),
    CoreTerm("吴昊", "人物"),
    CoreTerm("林焱", "人物"),
    CoreTerm("凤清儿", "人物"),
    CoreTerm("古青阳", "人物"),
    CoreTerm("魂灭生", "人物"),
    CoreTerm("玄空子", "人物"),
    CoreTerm("云岚宗", "势力"),
    CoreTerm("迦南学院", "势力"),
    CoreTerm("丹塔", "势力"),
    CoreTerm("魂殿", "势力"),
    CoreTerm("星陨阁", "势力"),
    CoreTerm("古族", "势力"),
    CoreTerm("魂族", "势力"),
    CoreTerm("太虚古龙", "族群"),
    CoreTerm("天妖凰族", "族群"),
    CoreTerm("加玛帝国", "地点"),
    CoreTerm("乌坦城", "地点"),
    CoreTerm("黑角域", "地点"),
    CoreTerm("中州", "地点"),
    CoreTerm("内院", "地点"),
    CoreTerm("异火", "设定"),
    CoreTerm("焚决", "功法"),
    CoreTerm("佛怒火莲", "斗技"),
    CoreTerm("陨落心炎", "异火"),
    CoreTerm("净莲妖火", "异火"),
    CoreTerm("三千焱炎火", "异火"),
    CoreTerm("青莲地心火", "异火"),
    CoreTerm("玄重尺", "物品"),
    CoreTerm("纳戒", "物品"),
    CoreTerm("厄难毒体", "体质"),
    CoreTerm("天焚炼气塔", "地点"),
)

TERM_BY_LABEL: Final = {term.label: term for term in CORE_TERMS}


@dataclass(frozen=True)
class KnownCoreFact:
    """A reviewed proposition tied to immutable formal-corpus line numbers."""

    fact_id: str
    entity: str
    canonical_question: str
    answer: str
    evidence_lines: tuple[int, ...]
    evidence_needles: tuple[tuple[str, ...], ...]
    required_terms: tuple[str, ...]
    acceptance_case_id: str = ""


KNOWN_CORE_FACTS: Final = (
    KnownCoreFact(
        "xiaoyan_identity",
        "萧炎",
        "萧炎是谁？",
        "萧炎是萧战的儿子；原文同时说明萧战是萧家现任族长。",
        (93,),
        (("萧炎", "父亲", "萧战", "萧家现任族长"),),
        ("萧炎", "萧战"),
        "known_core_xiaoyan",
    ),
    KnownCoreFact(
        "xiaozhan_identity",
        "萧战",
        "萧战是谁？",
        "萧战是萧家现任族长，也是萧炎的父亲。",
        (93,),
        (("萧家现任族长", "萧炎的父亲", "萧战"),),
        ("萧战", "萧炎", "族长"),
    ),
    KnownCoreFact(
        "yaochen_identity",
        "药尘",
        "药尘是谁？",
        "药尘是药老曾经的名字；药老是萧炎的老师。",
        (35484, 3861),
        (("药尘", "老师曾经的名字", "萧炎"), ("药老", "萧炎", "老师")),
        ("药尘", "药老", "老师"),
        "known_core_yaochen",
    ),
    KnownCoreFact(
        "yaolao_yaochen_alias",
        "药老",
        "药老和药尘是什么关系？",
        "药尘是药老曾经的名字，两者指向同一人物；药老也是萧炎的老师。",
        (35484, 3861),
        (("药尘", "老师曾经的名字", "萧炎"), ("药老", "萧炎", "老师")),
        ("药老", "药尘", "同一人物"),
        "known_core_yaolao_yaochen",
    ),
    KnownCoreFact(
        "yaolao_teacher",
        "药老",
        "药老和萧炎是什么关系？",
        "药老是萧炎的老师。",
        (3861,),
        (("药老", "萧炎", "老师"),),
        ("药老", "萧炎", "老师"),
    ),
    KnownCoreFact(
        "yunlanzong_identity",
        "云岚宗",
        "云岚宗是什么？",
        "云岚宗是加玛帝国境内极有影响力的势力，原文把它称作整个加玛帝国的一霸。",
        (237,),
        (("云岚宗", "加玛帝国", "一霸"),),
        ("云岚宗", "加玛帝国"),
        "known_core_yunlanzong",
    ),
    KnownCoreFact(
        "nalan_identity",
        "纳兰嫣然",
        "纳兰嫣然是谁？",
        "纳兰嫣然是纳兰桀的孙女，也是与萧炎指腹为婚的未婚妻。",
        (251,),
        (("纳兰桀的孙女", "纳兰嫣然", "未婚妻", "萧炎"),),
        ("纳兰嫣然", "纳兰桀", "萧炎"),
    ),
    KnownCoreFact(
        "yunyun_identity",
        "云韵",
        "云韵是谁？",
        "云韵是云岚宗宗主，也是纳兰嫣然的老师。",
        (257,),
        (("云岚宗宗主云韵", "纳兰嫣然", "弟子"),),
        ("云韵", "云岚宗", "纳兰嫣然"),
    ),
    KnownCoreFact(
        "yihuo_role",
        "异火",
        "异火在炼药中有什么作用？",
        "异火可以用于炼药；原文说明它能提高成功率，并增强丹药药效。",
        (393,),
        (("天地异火", "炼药", "成功率", "药效"),),
        ("异火", "炼药"),
    ),
    KnownCoreFact(
        "guhe_title",
        "古河",
        "古河有什么称号？",
        "古河被称为丹王。",
        (411,),
        (("丹王古河",),),
        ("古河", "丹王"),
    ),
    KnownCoreFact(
        "fanjue_identity",
        "焚决",
        "焚决是什么？",
        "焚决是一门功法；它能够进化，而进化需要吞噬异火。",
        (4768, 4770),
        (("功法", "焚决"), ("焚决", "进化", "异火", "吞噬")),
        ("焚决", "功法", "异火"),
    ),
    KnownCoreFact(
        "medusa_identity",
        "美杜莎",
        "美杜莎女王是什么身份？",
        "美杜莎女王是沙漠蛇族的皇者。",
        (15597,),
        (("蛇族的皇者", "美杜莎女王"),),
        ("美杜莎女王", "蛇族", "皇者"),
    ),
    KnownCoreFact(
        "haibodong_title",
        "海波东",
        "海波东有什么称号？",
        "海波东的称号是冰皇。",
        (15617,),
        (("冰皇海波东",),),
        ("海波东", "冰皇"),
    ),
    KnownCoreFact(
        "fire_lotus_creator",
        "佛怒火莲",
        "佛怒火莲是谁创造的？",
        "佛怒火莲是萧炎创造的招式。",
        (23382,),
        (("萧炎", "创造出来的", "佛怒火莲"),),
        ("佛怒火莲", "萧炎"),
    ),
    KnownCoreFact(
        "black_corner_identity",
        "黑角域",
        "黑角域是什么地方？",
        "黑角域是大陆上的特殊地域，那里没有通常的法律约束，奉行丛林法则。",
        (35197, 35201),
        (("特殊地域", "黑角域"), ("黑角域", "法律约束", "丛林法则")),
        ("黑角域", "特殊地域"),
    ),
    KnownCoreFact(
        "soul_hall_identity",
        "魂殿",
        "魂殿是什么？",
        "魂殿是遍布大陆多地的神秘势力。",
        (35488,),
        (("魂殿", "神秘势力", "大陆"),),
        ("魂殿", "神秘势力"),
    ),
    KnownCoreFact(
        "dan_tower_identity",
        "丹塔",
        "丹塔是什么？",
        "丹塔是一个受到许多炼药师推崇的自由组织。",
        (70306,),
        (("炼药师推崇", "丹塔", "自由组织"),),
        ("丹塔", "炼药师", "组织"),
    ),
    KnownCoreFact(
        "xingyun_pavilion_identity",
        "星陨阁",
        "星陨阁是什么势力？",
        "星陨阁是中州四阁之一，也是中州的一流势力。",
        (83273,),
        (("星陨阁", "中州", "一流势力"),),
        ("星陨阁", "中州", "势力"),
    ),
)


DIRECT_CORE_QUESTION_SUFFIXES: Final = (
    "请直接回答。",
    "用一句自然的话说明。",
    "只说有原著依据的部分。",
    "请给新读者简要解释。",
    "不补充无依据细节。",
    "请先给结论，再说明基本关系。",
    "怎样回答最准确？",
    "请用清楚、简洁的方式回答。",
    "不要回避这个核心已知问题。",
    "请陈述可核验的身份或关系。",
    "回答应包含关键专名。",
    "请避免把它与其他人物或势力混淆。",
    "用不超过三句话回答。",
    "请保持小说领域语境。",
    "直接说明核心信息即可。",
    "请给出明确而不含糊的结论。",
)

DIRECT_CORE_SPLIT_LEADS: Final = {
    "train": "{question}",
    "val": "读者想确认一件事：{question}",
    "public_diagnostic": "关于原著，{question}",
    "sealed_test": "有人询问小说设定：{question}",
}


def _bank(*templates: str) -> tuple[str, ...]:
    return tuple(templates)


# Every family has split-specific wording.  Template IDs are generated as
# ``{split}:{family}:p{index}``, and the builder additionally records the
# SHA-256 of the unrendered template text.
PROMPT_BANKS: Final = {
    "core_fact": {
        "train": _bank(
            "原著第{chapter_number}章写到{entity}时，原文明示了什么？",
            "请直接概括第{chapter_number}章中与{entity}有关的这条信息。",
            "关于{entity}，第{chapter_number}章留下了哪项明确事实？",
            "不补写情节，第{chapter_number}章怎样提及{entity}？",
        ),
        "val": _bank(
            "用一句自然的话说明第{chapter_number}章对{entity}的明确描写。",
            "第{chapter_number}章涉及{entity}，可核实的内容是什么？",
            "只依据原著，第{chapter_number}章给出了{entity}的什么信息？",
            "读完第{chapter_number}章，关于{entity}能确定哪一点？",
        ),
        "public_diagnostic": _bank(
            "读者追问第{chapter_number}章里的{entity}，应怎样准确作答？",
            "第{chapter_number}章与{entity}相关的事实，请直接说清楚。",
            "不添加后续剧情，原著在第{chapter_number}章怎样写{entity}？",
            "针对第{chapter_number}章，概括{entity}被写到的内容。",
        ),
        "sealed_test": _bank(
            "第{chapter_number}章出现{entity}时，文本明确表达了什么？",
            "请准确复述第{chapter_number}章关于{entity}的一项信息。",
            "原著第{chapter_number}章提到{entity}，哪些内容有据可查？",
            "面对关于第{chapter_number}章中{entity}的提问，怎样简要回答？",
        ),
    },
    "core_correction": {
        "train": _bank(
            "有人说第{chapter_number}章没有出现{entity}，请依据原著纠正。",
            "判断并修正：第{chapter_number}章与{entity}无关。",
            "“第{chapter_number}章未提到{entity}”可靠吗？请说明。",
        ),
        "val": _bank(
            "核对说法：原著第{chapter_number}章没有写到{entity}。",
            "若有人否认第{chapter_number}章提及{entity}，应如何回应？",
            "请校正“{entity}未在第{chapter_number}章出现”这一判断。",
        ),
        "public_diagnostic": _bank(
            "读者断言第{chapter_number}章找不到{entity}，该怎样核验？",
            "第{chapter_number}章是否真的与{entity}毫无关系？",
            "请用原著依据纠正关于第{chapter_number}章和{entity}的错误说法。",
        ),
        "sealed_test": _bank(
            "“第{chapter_number}章完全没写{entity}”是否成立？",
            "怎样反驳第{chapter_number}章不含{entity}这一说法？",
            "请核实第{chapter_number}章是否提到了{entity}。",
        ),
    },
    "passage_answer": {
        "train": _bank(
            "阅读片段：\n{quote}\n\n这段文字明确写到了什么？",
            "材料如下：\n{quote}\n\n请概括其中与{entity}有关的明确信息。",
            "依据这段原文回答，不扩展到别处：\n{quote}\n\n{entity}在此处怎样被提及？",
            "请从下列片段提取可核实的信息：\n{quote}\n\n回答应围绕{entity}。",
        ),
        "val": _bank(
            "原文片段：\n{quote}\n\n用自己的话说明它对{entity}表达了什么。",
            "看完下面材料，只陈述能被材料支持的内容：\n{quote}\n\n重点是{entity}。",
            "这段小说文字提供了什么与{entity}有关的线索？\n{quote}",
            "请解释下面片段中{entity}所处的局部情境：\n{quote}",
        ),
        "public_diagnostic": _bank(
            "给定证据：\n{quote}\n\n关于{entity}，哪些说法能由证据直接支持？",
            "请阅读证据并回答其中对{entity}的描写：\n{quote}",
            "不借助片段外信息，说明{entity}在下面文字中被怎样写到：\n{quote}",
            "从这段材料归纳一项可验证结论：\n{quote}\n\n对象是{entity}。",
        ),
        "sealed_test": _bank(
            "根据以下原著内容，准确说明{entity}相关的信息：\n{quote}",
            "这份证据能够支持关于{entity}的什么结论？\n{quote}",
            "只在给定文本范围内回答：\n{quote}\n\n问题围绕{entity}。",
            "读完片段后，怎样概括{entity}在此处的情况？\n{quote}",
        ),
    },
    "passage_insufficient": {
        "train": _bank(
            "片段如下：\n{quote}\n\n它能否证明{target}与{entity}存在明确关系？",
            "只看这段材料，能确认{target}的完整身份吗？\n{quote}",
            "有人据此断定{target}就是{entity}，证据够吗？\n{quote}",
        ),
        "val": _bank(
            "阅读材料后判断：它足以说明{target}和{entity}的关系吗？\n{quote}",
            "下面片段是否给出了{target}的确切身份？\n{quote}",
            "仅凭这段话，可以把{target}认作{entity}吗？\n{quote}",
        ),
        "public_diagnostic": _bank(
            "证据只有这一段：\n{quote}\n\n是否足以确认{target}与{entity}有关？",
            "这段原文能支持关于{target}的完整结论吗？\n{quote}",
            "请判断材料是否证明了{target}就是{entity}：\n{quote}",
        ),
        "sealed_test": _bank(
            "限定使用下列片段，它能证明{target}和{entity}的联系吗？\n{quote}",
            "关于{target}，这段证据是否已经充分？\n{quote}",
            "能否由下面文字推出{target}就是{entity}？\n{quote}",
        ),
    },
    "rag_compose": {
        "train": _bank(
            "综合以下证据片段，说明它们共同提供了什么信息：\n{bundle}\n\n回答围绕{entity}。",
            "把这些材料合并阅读，概括{entity}在其中的共同线索：\n{bundle}",
            "请区分证据与干扰内容，再说明{entity}可被确认的情况：\n{bundle}",
        ),
        "val": _bank(
            "阅读多段材料后，归纳与{entity}直接相关的结论：\n{bundle}",
            "哪些片段共同支持关于{entity}的判断？请综合作答。\n{bundle}",
            "合并下面证据，并排除无关信息：\n{bundle}\n\n对象：{entity}",
        ),
        "public_diagnostic": _bank(
            "这是一个检索结果包：\n{bundle}\n\n请形成关于{entity}的有据回答。",
            "综合这些检索片段，说明{entity}相关的共同信息：\n{bundle}",
            "从证据包中筛出有效内容并回答{entity}相关问题：\n{bundle}",
        ),
        "sealed_test": _bank(
            "依据下面的多段证据，对{entity}作出受支持的说明：\n{bundle}",
            "请综合证据包中真正涉及{entity}的内容：\n{bundle}",
            "多段材料中哪些内容能够共同说明{entity}？\n{bundle}",
        ),
    },
    "rag_insufficient": {
        split: _bank(
            wording[0], wording[1], wording[2]
        )
        for split, wording in {
            "train": (
                "这些片段能否证明{target}就是{entity}？请说明证据边界。\n{bundle}",
                "检索结果如下，它们足以确认{target}与{entity}的关系吗？\n{bundle}",
                "材料彼此分散，能否据此断定{target}属于{entity}？\n{bundle}",
            ),
            "val": (
                "综合这些片段后，是否能确认{target}和{entity}有关？\n{bundle}",
                "下面证据包足够支持关于{target}的结论吗？\n{bundle}",
                "能否把这些材料解释为{target}就是{entity}？\n{bundle}",
            ),
            "public_diagnostic": (
                "检索包是否真的证明了{target}与{entity}的联系？\n{bundle}",
                "仅凭以下多段材料，可否确认{target}的身份？\n{bundle}",
                "请检查证据是否足以推出{target}就是{entity}：\n{bundle}",
            ),
            "sealed_test": (
                "这些检索片段是否足以建立{target}和{entity}的关系？\n{bundle}",
                "请判断证据包能否确认{target}的完整身份：\n{bundle}",
                "多段文字能支持{target}就是{entity}这一结论吗？\n{bundle}",
            ),
        }.items()
    },
    "chat_first": {
        "train": _bank("我读到这段话：{quote}\n这里的{entity}该怎样理解？", "这段原文提到{entity}，你能自然解释一下吗？\n{quote}"),
        "val": _bank("看到下面这段时，我想弄清{entity}在这里的情况。\n{quote}", "请陪我读一下这段文字，重点讲{entity}。\n{quote}"),
        "public_diagnostic": _bank("读这段小说时，我对{entity}有点疑惑：\n{quote}", "这处情节写到{entity}，应怎样把握？\n{quote}"),
        "sealed_test": _bank("帮我理解下面片段里的{entity}：\n{quote}", "这里为什么会写到{entity}？请限定在片段内说明。\n{quote}"),
    },
    "chat_followup": {
        "train": _bank("你的说明依据原文中的哪一点？", "如果不看别处，这里还能确定什么？"),
        "val": _bank("哪些词句支撑了刚才的回答？", "这段材料没有说明的部分有哪些？"),
        "public_diagnostic": _bank("请指出刚才结论对应的文本依据。", "怎样避免把片段外情节混进回答？"),
        "sealed_test": _bank("刚才的解释能在材料中找到什么根据？", "哪些结论在这段文字之外，暂时不能下？"),
    },
    "chat_single": {
        "train": _bank("用自然语气谈谈这段文字中的{entity}：\n{quote}", "我只想了解当前片段里的{entity}，请简洁说明。\n{quote}"),
        "val": _bank("把下面关于{entity}的片段讲得容易理解一些：\n{quote}", "这段原文中的{entity}有什么值得注意的地方？\n{quote}"),
        "public_diagnostic": _bank("请像小说导读一样解释这段里的{entity}：\n{quote}", "面向新读者，怎样说明当前片段中的{entity}？\n{quote}"),
        "sealed_test": _bank("自然地介绍这段文字写到的{entity}：\n{quote}", "不剧透后文，解释片段中的{entity}。\n{quote}"),
    },
    "summary": {
        "train": _bank("用一两句话概括这段小说文字，不添加材料外信息：\n{quote}", "请提炼下面片段的局部要点：\n{quote}"),
        "val": _bank("把这段原文概括成简洁摘要：\n{quote}", "请总结片段正在写什么：\n{quote}"),
        "public_diagnostic": _bank("给下面的小说片段写一则短摘要：\n{quote}", "只保留关键信息，概括这段内容：\n{quote}"),
        "sealed_test": _bank("简要归纳以下片段表达的内容：\n{quote}", "请写出这段文字的局部梗概：\n{quote}"),
    },
    "rewrite": {
        "train": _bank("把下面原文改写成清楚的现代陈述，事实不变：\n{quote}", "请简洁改写这段话并保留{entity}：\n{quote}"),
        "val": _bank("用更直白的语言重述片段，不增添情节：\n{quote}", "将下列文字改写得紧凑一些，保留{entity}：\n{quote}"),
        "public_diagnostic": _bank("请在不改变事实的前提下改写：\n{quote}", "把这段小说文字转成简洁说明，必须保留{entity}：\n{quote}"),
        "sealed_test": _bank("重述下面片段，保持人物和事件信息不变：\n{quote}", "请用简明表达改写这段与{entity}有关的文字：\n{quote}"),
    },
    "continuation": {
        "train": _bank("原文片段如下：\n{quote}\n\n请接出紧随其后的短句，不自行续编。", "按原著补出这段之后的下一句：\n{quote}"),
        "val": _bank("请依据原文接写下一小段，不能自由创作：\n{quote}", "这段文字之后原著怎样继续？只给紧邻内容。\n{quote}"),
        "public_diagnostic": _bank("完成原著局部续接：\n{quote}\n\n输出下一句即可。", "不要发挥，补出下面片段后的直接后文：\n{quote}"),
        "sealed_test": _bank("请还原这段之后紧接的原文：\n{quote}", "局部接写下一句，答案必须来自原著：\n{quote}"),
    },
    "boundary_known": {
        "train": _bank("关于{entity}，当前模型能直接回答哪项有原著依据的信息？", "若读者问{entity}，请给出一项明确事实。"),
        "val": _bank("请直接回答一项关于{entity}的核心已知内容。", "对{entity}，哪些信息无需猜测即可回答？"),
        "public_diagnostic": _bank("读者询问{entity}，请给出有原著依据的简要回答。", "关于{entity}，请说出一项可核验事实。"),
        "sealed_test": _bank("面对{entity}的核心问题，应怎样直接回答？", "请陈述一项与{entity}有关且有据可查的信息。"),
    },
    "boundary_need_evidence": {
        "train": _bank("{entity}在全书首次出现于哪一章？", "{entity}在全书一共出现多少次？", "请给出{entity}贯穿全书的完整时间线。"),
        "val": _bank("能否直接断定{entity}首次登场的精确章节？", "请准确统计{entity}在全书的出现总数。", "{entity}的全书事件顺序是什么？"),
        "public_diagnostic": _bank("不使用索引时，能精确回答{entity}第一次出现的位置吗？", "没有逐章计数，能否给出{entity}的出现次数？", "请立即列出{entity}完整且无遗漏的全书时间线。"),
        "sealed_test": _bank("若没有检索结果，{entity}首次出现章节能确定吗？", "当前没有实体索引，能准确报出{entity}出现多少次吗？", "未提供逐章材料时，能否还原{entity}的完整时间线？"),
    },
    "boundary_resume": {
        "train": _bank("现在给出原文证据：\n{quote}\n\n基于它说明{entity}。", "已有材料如下，请恢复作答：\n{quote}\n\n问题围绕{entity}。"),
        "val": _bank("补充证据后，请说明其中的{entity}：\n{quote}", "这次提供了有效片段，请据此回答{entity}相关问题：\n{quote}"),
        "public_diagnostic": _bank("检索结果已经给出：\n{quote}\n\n请据此回答{entity}。", "材料已足够，请限定在下文说明{entity}：\n{quote}"),
        "sealed_test": _bank("以下证据可用，请据此恢复对{entity}的回答：\n{quote}", "已有原著片段，请只依据它解释{entity}：\n{quote}"),
    },
}


ANSWER_STYLE_BANKS: Final = {
    family: {
        split: tuple(f"{split}:{family}:a{index}" for index in range(6))
        for split in SPLITS
    }
    for family in PROMPT_BANKS
}


def prompt_template(family: str, split: str, index: int) -> tuple[str, str]:
    templates = PROMPT_BANKS[family][split]
    offset = index % len(templates)
    return f"{split}:{family}:p{offset}", templates[offset]


def answer_style_id(family: str, split: str, index: int) -> str:
    styles = ANSWER_STYLE_BANKS[family][split]
    return styles[index % len(styles)]
