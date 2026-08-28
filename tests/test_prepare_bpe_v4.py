import logging
import tempfile
import unittest
from pathlib import Path

from bpe_tokenizer import learn_bpe
from prepare_bpe_v4 import (
    SPECIAL_TOKENS,
    encode_with_eos,
    sha256_file,
    split_chapter_texts,
    write_manifest_sidecar,
)


class PrepareBPEV4Tests(unittest.TestCase):
    def test_manifest_sidecar_is_written_beside_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "isolated-output"
            output_dir.mkdir()
            manifest_path = output_dir / "token_manifest.json"
            manifest_path.write_text('{"status": "ready"}\n', encoding="utf-8")

            sidecar_path = write_manifest_sidecar(manifest_path)

            self.assertEqual(sidecar_path.parent, output_dir)
            self.assertEqual(
                sidecar_path.read_text(encoding="utf-8"),
                f"{sha256_file(manifest_path)}  token_manifest.json\n",
            )

    def test_chapter_encoding_appends_exactly_one_eos_per_chapter(self):
        text = (
            "------------\n\n第一章 开始\n\n天地玄黄。\n"
            "------------\n\n第二章 继续\n\n天地再会。\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.txt"
            path.write_text(text, encoding="utf-8")
            chapters = split_chapter_texts(path)
        tokenizer = learn_bpe(
            chapters,
            sorted(set(text)),
            num_merges=4,
        ).with_special_tokens(SPECIAL_TOKENS)
        eos_id = tokenizer.special_to_id["<EOS>"]
        tensor = encode_with_eos(
            chapters,
            tokenizer,
            eos_id,
            logging.getLogger("test.bpe_v4"),
            "train",
        )
        self.assertEqual(int((tensor == eos_id).sum()), 2)
        self.assertEqual(
            tokenizer.decode(tensor.tolist(), skip_special_tokens=True),
            text,
        )


if __name__ == "__main__":
    unittest.main()
