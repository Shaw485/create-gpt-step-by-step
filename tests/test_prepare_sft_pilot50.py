import unittest

from prepare_sft_pilot50 import validate_pilot_records


class PrepareSftPilotTests(unittest.TestCase):
    def make_records(self):
        records = []
        for split, count in (("train", 40), ("val", 5), ("test", 5)):
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

    def test_accepts_exact_40_5_5_split(self):
        records = self.make_records()
        chars = set("问trainvlest0123456789？答。证据")
        stoi = {char: index for index, char in enumerate(chars)}
        validate_pilot_records(records, "证据", stoi)

    def test_rejects_wrong_split_count(self):
        records = self.make_records()[:-1]
        chars = set("问trainvlest0123456789？答。证据")
        stoi = {char: index for index, char in enumerate(chars)}
        with self.assertRaisesRegex(ValueError, "expected split counts"):
            validate_pilot_records(records, "证据", stoi)

    def test_rejects_topic_leakage_into_test(self):
        records = self.make_records()
        records[-1]["topic"] = records[0]["topic"]
        chars = set("问trainvlest0123456789？答。证据")
        stoi = {char: index for index, char in enumerate(chars)}
        with self.assertRaisesRegex(ValueError, "topics leak"):
            validate_pilot_records(records, "证据", stoi)


if __name__ == "__main__":
    unittest.main()
