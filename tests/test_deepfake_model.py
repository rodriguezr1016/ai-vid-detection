from ai_video_detector.deepfake_model import DeepfakeModelClient


def test_parse_prediction_reads_mintime_stdout() -> None:
    client = DeepfakeModelClient(model_weights="/tmp/missing.ckpt")

    prediction = client._parse_prediction("Prediction 0.7321\n")

    assert prediction == 0.7321
