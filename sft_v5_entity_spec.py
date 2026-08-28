"""Shared factual cards and frozen entity diagnostics for SFT v5.2.

The facts in this module are intentionally short and conservative.  They are
already present in the curated project data; the module does not try to infer
new plot details from entity-name frequency alone.
"""

from __future__ import annotations


KNOWN_ENTITY_PROFILES = {
    "萧炎": {
        "kind": "person",
        "description": "是《斗破苍穹》的主要人物，故事主要围绕他展开",
        "required_context": ("主要", "人物", "主角", "核心"),
    },
    "药尘": {
        "kind": "person",
        "description": "又称药老，是萧炎的重要老师",
        "required_context": ("药老", "老师"),
    },
    "药老": {
        "kind": "person",
        "description": "就是药尘，也是萧炎的重要老师",
        "required_context": ("药尘", "老师"),
    },
    "异火": {
        "kind": "concept",
        "description": "是小说中的特殊火焰力量，和炼药、修炼有关",
        "required_context": ("火焰", "力量", "炼药"),
    },
    "韩枫": {
        "kind": "person",
        "description": "是小说中与药尘有关的重要人物",
        "required_context": ("药尘", "人物"),
    },
    "紫研": {
        "kind": "person",
        "description": "是《斗破苍穹》中的重要人物",
        "required_context": ("重要", "人物"),
    },
    "云韵": {
        "kind": "person",
        "description": "是《斗破苍穹》中的重要人物",
        "required_context": ("重要", "人物"),
    },
    "美杜莎": {
        "kind": "person",
        "description": "是《斗破苍穹》中的重要人物",
        "required_context": ("重要", "人物"),
    },
    "萧薰儿": {
        "kind": "person",
        "description": "是与萧炎关系密切的重要人物",
        "required_context": ("萧炎", "重要", "人物"),
    },
    "萧战": {
        "kind": "person",
        "description": "是萧炎的父亲",
        "required_context": ("萧炎", "父亲"),
    },
    "小医仙": {
        "kind": "person",
        "description": "是《斗破苍穹》中的重要人物",
        "required_context": ("重要", "人物"),
    },
    "海波东": {
        "kind": "person",
        "description": "是《斗破苍穹》中的重要人物",
        "required_context": ("重要", "人物"),
    },
    "纳兰嫣然": {
        "kind": "person",
        "description": "是《斗破苍穹》中的重要人物",
        "required_context": ("重要", "人物"),
    },
    "云山": {
        "kind": "person",
        "description": "是云岚宗的重要人物",
        "required_context": ("云岚宗", "重要", "人物"),
    },
    "古河": {
        "kind": "person",
        "description": "是小说中被称为丹王的人物",
        "required_context": ("丹王", "人物"),
    },
}

CORE_ENTITY_NAMES = ("萧炎", "药尘", "药老", "异火", "萧战")


PERSON_IDENTITY_QUESTIONS = (
    "小说里名为{name}的角色是什么身份？",
    "怎样概括{name}在故事中的身份？",
    "请说明{name}这个名字对应的人物定位。",
    "说说{name}在《斗破苍穹》中的基本定位。",
    "读到{name}时，应该先了解他的什么身份？",
    "用一句话概括{name}的小说身份。",
    "关于{name}，最基础的人物信息是什么？",
    "请先介绍{name}在故事里的身份。",
    "在这部小说中，{name}扮演什么角色？",
    "如果不了解{name}，应该先知道哪项基本信息？",
    "请简要说明{name}与这部故事的关系。",
    "只根据《斗破苍穹》，概括{name}的人物身份。",
)


CONCEPT_IDENTITY_QUESTIONS = (
    "小说里的{name}指哪一种力量？",
    "怎样概括{name}在故事中的含义？",
    "请说明{name}这个名称代表什么。",
    "说说{name}在《斗破苍穹》中的基本概念。",
    "读到{name}时，应该怎样理解它？",
    "用一句话概括{name}的小说设定。",
    "关于{name}，最基础的信息是什么？",
    "请先介绍{name}在故事里的作用类型。",
    "在这部小说中，{name}属于什么力量？",
    "如果不了解{name}，应该先知道哪项基本设定？",
    "请简要说明{name}与这部故事的关系。",
    "只根据《斗破苍穹》，概括{name}的含义。",
)


CORE_PERSON_EXTRA_QUESTIONS = (
    "请介绍{name}在小说里的基本身份。",
    "请简单介绍一下{name}这个角色。",
    "用一句话介绍一下{name}的身份。",
    "如果有人问{name}是哪位角色，应该怎样回答？",
    "{name}这个名字指的是哪位小说角色？",
    "小说中的{name}是哪一类人物？",
    "{name}在本书人物关系中处于什么位置？",
    "只说基本事实，{name}是怎样的角色？",
    "请用最基础的信息说明{name}的身份。",
    "第一次了解{name}时应先记住什么？",
    "这本书中的{name}可以怎样简要介绍？",
    "请给出{name}的人物身份结论。",
)


CORE_CONCEPT_EXTRA_QUESTIONS = (
    "请介绍{name}在小说里的基本设定。",
    "请简单介绍一下{name}这个概念。",
    "用一句话介绍一下{name}的含义。",
    "如果有人问{name}是哪类力量，应该怎样回答？",
    "{name}这个名称指的是哪种小说力量？",
    "小说中的{name}是哪一类设定？",
    "{name}在本书力量体系中属于什么？",
    "只说基本事实，{name}是怎样的力量？",
    "请用最基础的信息说明{name}的含义。",
    "第一次了解{name}时应先记住什么？",
    "这本书中的{name}可以怎样简要介绍？",
    "请给出{name}的设定结论。",
)


IDENTITY_ANSWERS = (
    "{name}{description}。",
    "在这部小说中，{name}{description}。",
    "简单概括，{name}{description}。",
)


RELATION_FACTS = (
    ("药老和药尘", "指的是同一个人物", "药老就是药尘"),
    ("药尘与萧炎", "是老师与学生的重要关系", "药尘是萧炎的重要老师"),
    ("药老与萧炎", "是老师与学生的重要关系", "药老是萧炎的重要老师"),
    ("萧战与萧炎", "是父亲与儿子的关系", "萧战是萧炎的父亲"),
    ("萧薰儿与萧炎", "关系密切", "萧薰儿是与萧炎关系密切的重要人物"),
    ("韩枫与药尘", "在人物关系上有关联", "韩枫是与药尘有关的重要人物"),
    ("古河与丹王", "是人物与称号的关系", "古河是小说中被称为丹王的人物"),
    ("云山与云岚宗", "是人物与宗门的重要关系", "云山是云岚宗的重要人物"),
)


UNKNOWN_ENTITY_NAMES = (
    "九星猫皇",
    "青木灵舟",
    "星河炼药塔",
    "北海药王",
    "玄风石碑",
    "风雪猫王",
    "青铜火莲",
    "天河药典",
    "南山火典",
    "白银斗帝",
    "赤月剑尊",
    "苍云圣塔",
    "玄海龙王",
    "七曜药宫",
    "碧落天舟",
    "九霄灵院",
    "黑曜古城",
    "星尘火皇",
    "寒川药典",
    "万木神殿",
)


HIDDEN_ENTITY_EVAL_ITEMS = (
    {
        "id": "hidden_xiaoyan_identity",
        "category": "已知实体",
        "question": "小说萧炎是谁？",
        "required_any": ("萧炎",),
        "required_context_any": ("主要", "人物", "主角", "核心"),
    },
    {
        "id": "hidden_xiaoyan_role",
        "category": "已知实体",
        "question": "不展开剧情，先说萧炎的基本角色定位。",
        "required_any": ("萧炎",),
        "required_context_any": ("主要", "人物", "主角", "核心"),
    },
    {
        "id": "hidden_yaochen_identity",
        "category": "已知实体",
        "question": "小说药尘是谁？",
        "required_any": ("药尘", "药老"),
        "required_context_any": ("药老", "老师"),
    },
    {
        "id": "hidden_yaochen_role",
        "category": "已知实体",
        "question": "药尘在萧炎成长中是什么身份？",
        "required_any": ("药尘", "药老"),
        "required_context_any": ("老师",),
    },
    {
        "id": "hidden_yaolao_alias",
        "category": "已知实体",
        "question": "大家所说的药老本名叫什么？",
        "required_any": ("药尘",),
        "required_context_any": ("药老", "药尘"),
    },
    {
        "id": "hidden_yihuo_concept",
        "category": "已知实体",
        "question": "异火在这部小说中属于哪类设定？",
        "required_any": ("异火",),
        "required_context_any": ("火焰", "力量", "炼药"),
    },
    {
        "id": "hidden_xiaozhan_relation",
        "category": "已知实体",
        "question": "萧战和萧炎是什么亲属关系？",
        "required_any": ("萧战", "萧炎"),
        "required_context_any": ("父亲", "父子"),
    },
    {
        "id": "hidden_guhename",
        "category": "已知实体",
        "question": "小说里被称为丹王的人物叫什么？",
        "required_any": ("古河",),
        "required_context_any": ("丹王",),
    },
    {
        "id": "hidden_yunshan",
        "category": "已知实体",
        "question": "云山主要和哪个宗门有关？",
        "required_any": ("云山",),
        "required_context_any": ("云岚宗",),
    },
    {
        "id": "hidden_hanfeng",
        "category": "已知实体",
        "question": "韩枫与哪位老师辈人物有关？",
        "required_any": ("韩枫", "药尘", "药老"),
        "required_context_any": ("药尘", "药老"),
    },
    *(
        {
            "id": f"hidden_unknown_{index:02d}",
            "category": "不存在实体",
            "question": f"只按本项目语料判断，书中能确认{name}的身份吗？",
            "entity": name,
        }
        for index, name in enumerate(UNKNOWN_ENTITY_NAMES[:10], 1)
    ),
)
