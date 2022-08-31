import numpy as np
import torch
import onnxruntime
import cv2

from model_lib.MiniFASNet import MiniFASNetV2, MiniFASNetV1SE

fas2_model_1 = '../../face_model/fas_silent/2.7_80x80_MiniFASNetV2.pth'
fas2_model_2 = '../../face_model/fas_silent/4_0_0_80x80_MiniFASNetV1SE.pth'



def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image = cv2.imread("../../yhfacelib/data/test/a1.png")
img = cv2.resize(image, (80, 80))
print(img.shape)

img = torch.from_numpy(img.transpose((2, 0, 1)))
img = img.float()
img = img.unsqueeze(0).to(device)

# test run onnx model
onnx_session= onnxruntime.InferenceSession("outputs/2.7_80x80_MiniFASNetV2.onnx")
onnx_inputs= {onnx_session.get_inputs()[0].name: to_numpy(img)}
onnx_output = onnx_session.run(None, onnx_inputs)

print(onnx_output)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

data_numpy = softmax(onnx_output[0][0])
print(data_numpy)
