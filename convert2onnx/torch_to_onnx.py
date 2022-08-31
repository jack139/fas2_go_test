import torch
import onnxruntime as onnxrt

from model_lib.MiniFASNet import MiniFASNetV2, MiniFASNetV1SE

fas2_model_1 = '../../face_model/fas_silent/2.7_80x80_MiniFASNetV2.pth'
fas2_model_2 = '../../face_model/fas_silent/4_0_0_80x80_MiniFASNetV1SE.pth'

h_input, w_input = 80, 80


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

kernel_size = get_kernel(h_input, w_input)


def load_weights(model, weights_path):
    state_dict = torch.load(weights_path, map_location=device)
    keys = iter(state_dict)
    first_layer_name = keys.__next__()
    if first_layer_name.find('module.') >= 0:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            name_key = key[7:]
            new_state_dict[name_key] = value
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)


model1 = MiniFASNetV2(conv6_kernel=kernel_size).to(device)
load_weights(model1, fas2_model_1)

model2 = MiniFASNetV1SE(conv6_kernel=kernel_size).to(device)
load_weights(model2, fas2_model_2)

model1.eval()
model2.eval()


# random input tensor
dummy_input = torch.randn(1, 3, 80, 80)

input_names = [ "actual_input" ]
output_names = [ "output" ]

# convert to onnx model
torch.onnx.export(model1, 
                  dummy_input,
                  "outputs/2.7_80x80_MiniFASNetV2.onnx",
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=True,
                  )


torch.onnx.export(model2, 
                  dummy_input,
                  "outputs/4_0_0_80x80_MiniFASNetV1SE.onnx",
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=True,
                  )

