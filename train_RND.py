import tqdm
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from net import RNDPolicy2
import sys
sys.path.append('/home/chenxu/Downloads/TRI_chen/UQ_baselines')
import data_loader
from argparse import ArgumentParser
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

'''
	Example usage:
	export CUDA_VISIBLE_DEVICES=0
	nohup python train_RND.py --square 1  > square_train.out
	export CUDA_VISIBLE_DEVICES=1
	nohup python train_RND.py --square 0  > transport_train.out
'''


parser = ArgumentParser()
parser.add_argument("--square", default=1, type=int)
args = parser.parse_args()
if __name__ == "__main__":
    square = args.square == 1
    X, Y = data_loader.get_data(square=square)
    input_dim = X.shape[-1]; full_in_dim = X.shape[1] * X.shape[-1]; output_dim = Y.shape[-1]
    train_data = torch.utils.data.TensorDataset(X)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    ckpt_file = 'square.ckpt' if square else 'transport.ckpt'
    EPOCHS = 200

    # choice of model/method
    net = RNDPolicy2(input_dim).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    if os.path.exists(ckpt_file):
        ckpt = torch.load(ckpt_file)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        starting_epochs = ckpt['epoch']
        losses = ckpt['losses']
    else:
        starting_epochs = 0
        losses = []

    t = tqdm.trange(starting_epochs, EPOCHS)
    for i in t:
        # Save checkpoint before to avoid nan loss
        ckpt = {
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': i+1,
            'losses': losses
        }
        torch.save(ckpt, ckpt_file)

        net.train()
        loss_i = []
        for x_batch in tqdm.tqdm(train_loader, desc='Training Batches'):
            observation = x_batch[0].to(device)
            optimizer.zero_grad()
            loss = net(observation).mean()
            loss.backward()
            # Terminate if loss is NaN
            if torch.isnan(loss):
                print("Loss is NaN")
                raise ValueError(f"NaN loss at epoch {i}")
            loss_i += [loss.item()]
            optimizer.step()
        """ Validation
        """
        t.set_description(f"val. loss: {loss.detach().cpu().numpy():.2f}")
        t.refresh()
        losses += [np.mean(loss_i)]

        plt.title(r"Training loss")
        plt.plot(losses)
        plt.show()
        suffix = 'square' if square == 1 else 'transport'
        os.makedirs('images', exist_ok=True)
        plt.savefig(f"images/training_loss_{suffix}.png")
        plt.clf()