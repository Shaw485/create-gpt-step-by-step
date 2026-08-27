from pathlib import Path
import hashlib
import json
import logging
import os
from logging.handlers import RotatingFileHandler


RAW_PATH = Path("data/raw/doupo_raw.txt")
CLEAN_PATH = Path("data/clean/doupo_stage1.txt")
REPORT_PATH = Path("data/clean/stage1_report.json")
LOG_PATH = Path("logs/clean_corpus.log")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def configure_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger("corpus_cleaner")
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.propagate = False

    io_level = os.getenv("CLEAN_IO_LOG_LEVEL", "INFO").upper()
    transform_level = os.getenv("CLEAN_TRANSFORM_LOG_LEVEL", "INFO").upper()
    logging.getLogger("corpus_cleaner.io").setLevel(io_level)
    logging.getLogger("corpus_cleaner.transform").setLevel(transform_level)


def clean_text(text):
    transform_logger = logging.getLogger("corpus_cleaner.transform")

    nul_count = text.count("\x00")
    crlf_count = text.count("\r\n")
    standalone_cr_count = text.count("\r") - crlf_count

    cleaned_text = text.replace("\r\n", "\n")
    cleaned_text = cleaned_text.replace("\r", "\n")
    cleaned_text = cleaned_text.replace("\x00", "")

    transform_logger.debug(
        "character counts before=%d after=%d",
        len(text),
        len(cleaned_text),
    )
    transform_logger.info(
        "normalized text nul_removed=%d crlf_normalized=%d cr_normalized=%d",
        nul_count,
        crlf_count,
        standalone_cr_count,
    )

    statistics = {
        "nul_removed": nul_count,
        "crlf_normalized": crlf_count,
        "standalone_cr_normalized": standalone_cr_count,
    }
    return cleaned_text, statistics


def main():
    configure_logging()
    io_logger = logging.getLogger("corpus_cleaner.io")

    try:
        io_logger.info("reading input path=%s", RAW_PATH)
        raw_bytes = RAW_PATH.read_bytes()
        io_logger.debug(
            "input read bytes=%d sha256=%s",
            len(raw_bytes),
            sha256_bytes(raw_bytes),
        )
        had_utf8_bom = raw_bytes.startswith(b"\xef\xbb\xbf")
        original_text = raw_bytes.decode("utf-8-sig")

        cleaned_text, statistics = clean_text(original_text)
        cleaned_bytes = cleaned_text.encode("utf-8")

        if b"\x00" in cleaned_bytes:
            raise RuntimeError("cleaned output still contains NUL bytes")

        CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        CLEAN_PATH.write_bytes(cleaned_bytes)

        report = {
            "input_path": str(RAW_PATH),
            "output_path": str(CLEAN_PATH),
            "input_sha256": sha256_bytes(raw_bytes),
            "output_sha256": sha256_bytes(cleaned_bytes),
            "input_bytes": len(raw_bytes),
            "output_bytes": len(cleaned_bytes),
            "input_characters": len(original_text),
            "output_characters": len(cleaned_text),
            "utf8_bom_removed": had_utf8_bom,
            "replacement_characters": cleaned_text.count("\ufffd"),
            "output_lines": cleaned_text.count("\n") + 1,
            **statistics,
        }
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        io_logger.info(
            "wrote output path=%s bytes=%d report=%s",
            CLEAN_PATH,
            len(cleaned_bytes),
            REPORT_PATH,
        )
    except Exception:
        io_logger.exception("stage1 cleaning failed input=%s", RAW_PATH)
        raise


if __name__ == "__main__":
    main()
