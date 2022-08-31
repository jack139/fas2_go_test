import torch
import torch.nn.functional as F
import cv2

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

#model2 = MiniFASNetV1SE(conv6_kernel=kernel_size).to(device)
#load_weights(model2, fas2_model_2)

model1.eval()
#model2.eval()



def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


image = cv2.imread("../../yhfacelib/data/test/a1.png")
img = cv2.resize(image, (80, 80))
print(img.shape)

img = torch.from_numpy(img.transpose((2, 0, 1)))
img = img.float()
img = img.unsqueeze(0).to(device)
print(img.shape)

with torch.no_grad():
    result = model1.forward(img)
    print(result)
    result = F.softmax(result).cpu().numpy()
    print(result)
