from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="cesaraha/yolov8s-surgical-instrument-detection-cholec80",
    filename="yolov8s_cholec80.pt"
)

print("Model downloaded to:")
print(model_path)