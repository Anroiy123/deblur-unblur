from scripts.run_restormer import select_output_file


def test_select_output_file_ignores_resized_input_in_result_directory(tmp_path):
    resized_input = tmp_path / "document.jpg"
    resized_input.write_bytes(b"resized input")

    model_output = tmp_path / "Motion_Deblurring" / "document.png"
    model_output.parent.mkdir()
    model_output.write_bytes(b"restormer output")

    assert select_output_file(tmp_path, resized_input) == model_output
