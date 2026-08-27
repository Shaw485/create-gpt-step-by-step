import unittest

from clean_content import clean_lines


class CleanContentTests(unittest.TestCase):
    def test_removes_whole_advertisement_line(self):
        lines = [
            "声明:本书由八零电子书(www.txt80.com)整理。",
            "这是一行正常正文。",
        ]

        output_lines, statistics = clean_lines(lines)

        self.assertEqual(output_lines, ["这是一行正常正文。"])
        self.assertEqual(statistics["whole_lines_removed"], 1)

    def test_removes_embedded_watermark(self):
        lines = ["萧炎一笑，八度吧首发继续前行。"]

        output_lines, statistics = clean_lines(lines)

        self.assertEqual(output_lines, ["萧炎一笑，继续前行。"])
        self.assertEqual(
            statistics["inline_replacement_counts"]["badu_watermark"],
            1,
        )

    def test_removes_embedded_update_banner(self):
        lines = [
            "正文结束。为了方便访问,请牢记天天中文网您的支持是我们最大的动力！"
        ]

        output_lines, statistics = clean_lines(lines)

        self.assertEqual(output_lines, ["正文结束。"])
        self.assertEqual(
            statistics["inline_replacement_counts"]["update_banner"],
            1,
        )

    def test_preserves_ordinary_text(self):
        lines = ["小猫沿着原路回到了家。"]

        output_lines, statistics = clean_lines(lines)

        self.assertEqual(output_lines, lines)
        self.assertEqual(statistics["whole_lines_removed"], 0)
        self.assertEqual(statistics["unresolved_candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
