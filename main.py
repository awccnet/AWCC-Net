import argparse
import os

import numpy as np
import torch
from datasets.awcc_dataset import Crowd
from models.CC import CrowdCounter
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', default='/datasets/JHU', help='testing data directory')
    parser.add_argument('--checkpoint', default='checkpoints/best.pth', help='checkpoint path')

    args = parser.parse_args()
    return args


@torch.no_grad()
def eval(model, dataloader):
    model.eval()  # Set model to evaluate mode
    epoch_res = []
    # Iterate over data.
    for inputs, count in tqdm(dataloader):
        inputs = inputs.to('cuda')

        assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        outputs = model.test_forward(inputs)
        err = count[0].item() - torch.sum(outputs).item()
        epoch_res.append(err)

    epoch_res = np.array(epoch_res)
    mse = np.sqrt(np.mean(np.square(epoch_res)))
    mae = np.mean(np.abs(epoch_res))
    print('mse: {:.2f}, mae: {:.2f}'.format(mse, mae))


if __name__ == '__main__':
    args = parse_args()

    # prepare data
    dataroot = args.dir_path
    dataset = Crowd(os.path.join(dataroot, 'test'), 512, 16, 'val')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    # prepare model
    model = CrowdCounter(args)
    ckpt_path = args.checkpoint
    ckpt = torch.load(ckpt_path)['state_dict']
    msg = model.load_state_dict(ckpt)
    print(msg)
    model.cuda()

    # eval
    eval(model, dataloader)