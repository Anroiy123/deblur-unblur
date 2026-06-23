from pathlib import Path

import create_report


def test_help_text_is_compatible_with_windows_cp1258():
    help_text = create_report.build_parser().format_help()

    help_text.encode("cp1258")


def test_parse_args_uses_portable_default_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    args = create_report.parse_args(["reference.docx"])

    assert args.reference == Path("reference.docx")
    assert args.output == tmp_path / "BaoCao_DeBlur_analysis.txt"


def test_parse_args_accepts_explicit_output():
    args = create_report.parse_args(
        ["reference.docx", "--output", "reports/analysis.txt"]
    )

    assert args.output == Path("reports/analysis.txt")
