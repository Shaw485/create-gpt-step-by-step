import unittest

from audit_chapter_versions import audit_versions, compare_sections


def make_section(chapter_number, start_line, body_lines):
    body_characters = sum(len(line) for line in body_lines)
    return {
        "chapter_number": chapter_number,
        "title": f"第{chapter_number}章 测试",
        "start_line": start_line,
        "end_line": start_line + len(body_lines),
        "body_lines": body_lines,
        "body_characters": body_characters,
        "noise_hits": 0,
        "internal_heading_count": 0,
        "quality_score": body_characters,
    }


class ChapterVersionTests(unittest.TestCase):
    def setUp(self):
        self.shared_lines = [f"共同正文段落{i}" * 20 for i in range(10)]

    def test_exact_containment_has_full_coverage(self):
        shorter = make_section(100, 10, self.shared_lines)
        longer = make_section(100, 100, self.shared_lines + ["额外完整正文" * 30])

        similarity = compare_sections(shorter, longer)

        self.assertEqual(similarity["shared_coverage"], 1.0)

    def test_same_chapter_containment_is_high_confidence(self):
        shorter = make_section(100, 10, self.shared_lines)
        longer = make_section(100, 100, self.shared_lines + ["额外完整正文" * 30])

        pairs = audit_versions([shorter, longer], [(0, 1)])

        self.assertEqual(pairs[0]["confidence"], "high")
        self.assertEqual(pairs[0]["recommended_keep_start_line"], 100)

    def test_different_chapters_require_review(self):
        first = make_section(100, 10, self.shared_lines)
        second = make_section(101, 100, self.shared_lines)

        pairs = audit_versions([first, second], [(0, 1)])

        self.assertEqual(pairs[0]["confidence"], "review")

    def test_unrelated_sections_are_not_versions(self):
        first = make_section(100, 10, ["甲" * 600])
        second = make_section(100, 100, ["乙" * 600])

        pairs = audit_versions([first, second], [(0, 1)])

        self.assertEqual(pairs, [])


if __name__ == "__main__":
    unittest.main()
