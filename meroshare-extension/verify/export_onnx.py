"""Export the trained DigitCNN to ONNX for in-browser inference (onnxruntime-web)."""
import sys, os
sys.path.insert(0, "D:/meroshare")          # import the model definition from train.py
import torch
import train

OUT = "D:/meroshare/meroshare-extension/src/digit_cnn.onnx"

model = train.DigitCNN()
model.load_state_dict(torch.load("D:/meroshare/digit_cnn.pth", map_location="cpu"))
model.eval()

dummy = torch.randn(1, 2, train.CROP_SIZE, train.CROP_SIZE)   # [N, 2ch, 28, 28]
torch.onnx.export(
    model, dummy, OUT,
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=11,
    dynamo=False,          # force the legacy TorchScript exporter (no onnxscript dep)
)
print("exported", OUT, os.path.getsize(OUT), "bytes")
