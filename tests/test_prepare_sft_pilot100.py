import unittest

from prepare_sft_pilot100 import validate_expanded_records


class PrepareSftPilot100Tests(unittest.TestCase):
    def make_records(self):
        records = []
        for split, count in (("train", 80), ("val", 15), ("test", 5)):
            for index in range(count):
                records.append(
                    {
                        "id": f"{split}_{index}",
                        "question": f"问{split}{index}？",
                        "answer": "答。",
                        "evidence": "证据",
                        "source_line": 1,
                        "topic": f"{split}_{index}",
                        "split": split,
                    }
                )
        return records

    def test_accepts_80_15_5_split(self):
        records = self.make_records()
        chars = set("问trainvlest0123456789？答。证据")
        stoi = {char: index for index, char in enumerate(chars)}
        validate_expanded_records(records, "证据", stoi)

    def test_rejects_duplicate_question(self):
        records = self.make_records()
        records[-1]["question"] = records[0]["question"]
        chars = set("问trainvlest0123456789？答。证据")
        stoi = {char: index for index, char in enumerate(chars)}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_expanded_records(records, "证据", stoi)

    def test_rejects_wrong_counts(self):
        records = self.make_records()[:-1]
        chars = set("问trainvlest0123456789？答。证据")
        stoi = {char: index for index, char in enumerate(chars)}
        with self.assertRaisesRegex(ValueError, "unexpected split counts"):
            validate_expanded_records(records, "证据", stoi)


if __name__ == "__main__":
    unittest.main()
