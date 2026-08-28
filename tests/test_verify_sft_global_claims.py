import unittest

from prepare_corpus_v4 import Chapter
from verify_sft_global_claims import (
    ChapterTextIndex,
    count_support,
    extract_chapter_numbers,
    verify_record,
)


def chapter(number, text, index=0):
    return Chapter(
        section_id=f"chapter-{number}-{index}",
        section_index=index,
        chapter_number=number,
        title=f"第{number}章 测试",
        start_line=1,
        end_line=3,
        title_offset=1,
        source_text=f"-----\n第{number}章 测试\n{text}\n",
    )


class GlobalClaimVerificationTests(unittest.TestCase):
    def test_first_occurrence_uses_narrative_order(self):
        index = ChapterTextIndex(
            [chapter(2, "青鳞出现。", 0), chapter(1, "还没有这个人。", 1)]
        )
        self.assertEqual(index.first_chapter("青鳞"), 2)

    def test_late_sequel_chapter_one_does_not_become_first_occurrence(self):
        index = ChapterTextIndex(
            [chapter(100, "古元首次出现。", 0), chapter(1, "古元再次出现。", 1)]
        )
        self.assertEqual(index.first_chapter("古元"), 100)

    def test_first_cooccurrence_requires_same_retained_version(self):
        index = ChapterTextIndex(
            [
                chapter(1, "萧炎出现。", 0),
                chapter(1, "药老出现。", 1),
                chapter(2, "萧炎与药老同时出现。", 2),
            ]
        )
        self.assertEqual(index.first_cooccurrence(["萧炎", "药老"]), 2)

    def test_count_support_checks_each_retained_version(self):
        index = ChapterTextIndex(
            [
                chapter(1, "萧炎萧炎薰儿", 0),
                chapter(1, "萧炎薰儿薰儿", 1),
            ]
        )
        support = count_support(index, 1, "薰儿", set(), {"萧炎", "薰儿"})
        self.assertEqual(support["status"], "literal_count_mixed_or_missing_versions")

    def test_extract_chapter_numbers_supports_chinese_and_arabic(self):
        self.assertEqual(extract_chapter_numbers("第十二章和第20章"), [12, 20])

    def test_short_appearance_order_answer_can_be_index_verified(self):
        index = ChapterTextIndex(
            [chapter(1, "韩月出现。", 0), chapter(2, "雷尊者出现。", 1)]
        )
        record = {
            "id": "test",
            "split": "test",
            "question": "谁更早？",
            "answer": "韩月更早。",
            "origin": {
                "source_subcategory": "appearance_order",
                "source_chapter_number": 1,
                "source_entities": ["韩月", "雷尊者"],
            },
        }
        self.assertEqual(
            verify_record(record, index, {"韩月", "雷尊者"})["status"],
            "index_verified",
        )


if __name__ == "__main__":
    unittest.main()
