import torch.nn as nn
import torch

class CrowdCounter(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.device = torch.device('cuda')
        # model
        from models.AWCC import vgg19_trans  
        self.net = vgg19_trans(use_pe=True)

    
    @torch.no_grad()
    def test_forward(self, x, **kwargs):
        input_list = [x]
        dmap_list = []
        for input in input_list:
            pred_map, _ = self.net(input)
            dmap_list.append(pred_map.detach())
        return torch.relu(dmap_list[0])

