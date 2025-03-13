import torch

model = torch.load('./models/geom_difflinker.ckpt', map_location='cpu')

model_state = model['state_dict']

state_dict = {}

for key, val in model_state.items():
    k = key.split('.')
    if k[0] == 'edm':
        k = k[1:]
        k = '.'.join(k)
        state_dict[k] = val

torch.save(state_dict, './models/geom_diffliner_state_dict.pt')