import torch
import pickle
from collections import OrderedDict

# original saved file with DataParallel
checkpoint = torch.load('/mnt/datagrid/personal/paplhjak/BThesis/data/saved/models/Kitti_24_06/0624_110039/checkpoint-epoch55.pth', map_location=torch.device('cpu'))
state_dict = checkpoint['state_dict']
# create new OrderedDict that does not contain `module.`
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    print(k[:7])
    if k[:7]=='module.': 
      name = k[7:] # remove `module.`
      new_state_dict[name] = v
    else:
      new_state_dict[k] = v

checkpoint['state_dict'] = new_state_dict
checkpoint['monitor_best'] = 100

torch.save(checkpoint, "/mnt/datagrid/personal/paplhjak/BThesis/data/saved/models/Kitti_24_06/0624_110039/checkpoint-epoch55-single-gpu.pth")