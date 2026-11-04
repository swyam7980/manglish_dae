"""The second-stage wrapper and its results table. CPU only, no downloads, no model."""


def test_translate_batches_by_length_and_keeps_input_order():
    from translators import Translator
    tr = Translator.__new__(Translator)
    seen = []
    tr._chunk = lambda chunk: seen.extend(chunk) or [s.upper() for s in chunk]
    assert tr.translate(["ccc", "a", "", "bb"], batch=2) == ["CCC", "A", ".", "BB"]  # blank lines stay valid input
    assert [len(s) for s in seen] == sorted(len(s) for s in seen)


def test_markdown_lists_only_scored_inputs(tmp_path):
    from eval_translators import write_markdown
    path = tmp_path / "t.md"
    write_markdown({"in22_conv": {"it2_1b": {"gold": {"BLEU": 31.2, "chrF": 55.4}}}}, path)
    text = path.read_text(encoding="utf-8")
    assert "| it2_1b | MIT | 31.2 / 55.4 | - |" in text and "in22_gen" not in text
