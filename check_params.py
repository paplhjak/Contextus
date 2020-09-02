import torch
import io

with open("model_best.pth", 'rb') as f:
        buffer = io.BytesIO(f.read())

checkpoint = torch.load(buffer, map_location=torch.device('cpu'))

data = []
#checkpoint = torch.load("model_best.pth", map_location=torch.device('cpu'))
for param_tensor in checkpoint['state_dict']:
    #print(param_tensor, "\t", checkpoint['state_dict'][param_tensor].size())
    if 'sigma' in param_tensor:
        line_new = '{:<60}  {:>12}'.format(param_tensor, str(checkpoint['state_dict'][param_tensor].item()))
        print(line_new)
        #data.append(checkpoint['state_dict'][param_tensor].item())
        