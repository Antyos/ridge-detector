from pathlib import Path

from ridge_detector.detector import RidgeDetector, RidgeDetectorConfig

sample_parameters = {
    "img0.jpg": RidgeDetectorConfig(
        line_widths=13,
        low_contrast=64,
        high_contrast=229,
        min_len=50,
        max_len=0,
        dark_line=False,
        estimate_width=True,
        extend_line=False,
        correct_pos=False,
    ),
    "img1.jpg": RidgeDetectorConfig(
        line_widths=[1, 3],
        low_contrast=131,
        high_contrast=200,
        min_len=12,
        max_len=0,
        dark_line=True,
        estimate_width=True,
        extend_line=True,
        correct_pos=True,
    ),
    "img2.jpg": RidgeDetectorConfig(
        line_widths=10,
        low_contrast=11,
        high_contrast=189,
        min_len=5,
        max_len=0,
        dark_line=False,
        estimate_width=True,
        extend_line=True,
        correct_pos=False,
    ),
    "img3.jpg": RidgeDetectorConfig(),  # TODO
    "img4.jpg": RidgeDetectorConfig(),  # TODO
    "img5.png": RidgeDetectorConfig(),  # TODO
    "img6.jpg": RidgeDetectorConfig(),  # TODO
    "img7.png": RidgeDetectorConfig(),  # TODO
}


def generate_samples():
    data_dir = Path(__file__).parent.parent / "data"
    input_dir = data_dir / "images"
    out_dir = data_dir / "results"
    for img_name, params in sample_parameters.items():
        print(f"Processing {img_name} with config: {params}")
        file_path = input_dir / img_name
        detector = RidgeDetector(params)
        result = detector.detect_lines(file_path)
        result.export_images(
            out_dir,
            prefix=file_path.stem,
            draw_width=True,
            make_binary=True,
        )
    print("Done!")


if __name__ == "__main__":
    generate_samples()
