import unittest

from clean_duplicate_versions import build_remove_ranges, merge_overlapping_ranges, remove_ranges


def make_pair(section_one, section_two, keep, remove):
    first = section_one.copy()
    second = section_two.copy()
    return {
        "confidence": "high",
        "same_chapter_number": True,
        "first": first,
        "second": second,
        "matched_characters": min(
            section_one["body_characters"],
            section_two["body_characters"],
        ),
        "shared_coverage": 1.0,
        "recommended_keep_start_line": keep["start_line"],
        "recommended_remove_start_line": remove["start_line"],
    }


class DuplicateVersionCleanerTests(unittest.TestCase):
    def test_remove_previous_separator_to_next_separator(self):
        lines = [
            "------------\n",
            "第一千零一章 一场大战\n",
            "正文 A\n",
            "------------\n",
            "第一千零一章 一场大战\n",
            "正文 B\n",
            "------------\n",
        ]

        section_a = {
            "chapter_number": 1001,
            "title": "第一千零一章 一场大战",
            "start_line": 2,
            "end_line": 3,
            "body_characters": 7,
        }
        section_b = {
            "chapter_number": 1001,
            "title": "第一千零一章 一场大战",
            "start_line": 5,
            "end_line": 7,
            "body_characters": 7,
        }
        pair = make_pair(section_a, section_b, keep=section_a, remove=section_b)
        ranges = build_remove_ranges(lines, [pair])
        kept_text, removed_lines, removed_chars, removed_sections = remove_ranges(lines, ranges)

        self.assertEqual(len(ranges), 1)
        self.assertEqual(ranges[0]["line_range"], [3, 7])
        self.assertIn("正文 A", kept_text)
        self.assertNotIn("正文 B", kept_text)
        self.assertEqual(removed_sections[0]["line_count"], 4)
        self.assertGreater(removed_chars, 0)
        self.assertEqual(removed_lines, 4)

    def test_overlap_ranges_raise(self):
        lines = ["正文\n"] * 20
        ranges = [
            {"line_range": [1, 10], "chapter_number": 1, "title": "a",
             "matched_characters": 0, "shared_coverage": 1.0, "recommended_keep_start_line": 1},
            {"line_range": [8, 15], "chapter_number": 2, "title": "b",
             "matched_characters": 0, "shared_coverage": 1.0, "recommended_keep_start_line": 1},
        ]
        with self.assertRaises(ValueError):
            merge_overlapping_ranges(ranges)


if __name__ == "__main__":
    unittest.main()
