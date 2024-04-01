from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor, Lambda,Resize
from torchvision.datasets.mnist import MNIST, FashionMNIST
import argparse,random
from network.DDPM import*
from utils import*

parser = argparse.ArgumentParser(description='Training')
# model parameters
parser.add_argument("--size", type=int, default=32)
parser.add_argument("--hidden_channels", type=int, default=64)
parser.add_argument("--time_channels", type=int, default=512)
parser.add_argument("-K", "--num_blocks", type=int, default=8)
parser.add_argument("-L", "--num_groups", type=int, default=32)
parser.add_argument("-N", "--attention", type=int, default=16)
parser.add_argument("-T", "--time_steps", type=int, default=1000)
parser.add_argument( "--beta_min", type=int, default=0.0001)
parser.add_argument( "--beta_max", type=int, default=0.02)


#dataset
parser.add_argument("--fashion", type=bool, default=False)

# Optimizer parameters
parser.add_argument("--lr", type=float, default=0.0002)
# Trainer parameters
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_steps", type=int, default=10000)

args = parser.parse_args()

# Setting reproducibility
SEED = 126
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Definitions
STORE_PATH_MNIST = f"ddpm_model_mnist.pt"
STORE_PATH_FASHION = f"ddpm_model_fashion.pt"

fashion = False
batch_size = args.batch_size
lr = args.lr
img_size = args.size
channels = 1
hidden = args.hidden_channels
hidden_mult = [1,2,4,8]
time_channels = args.time_channels
num_groups = args.num_groups
resolution_attention = args.attention
num_res_blocks = args.num_blocks
max_steps = args.time_steps
min_beta=args.beta_min
max_beta=args.beta_max
device = 'cuda' if torch.cuda.is_available() else 'cpu'
store_path = "ddpm_fashion.pt" if fashion else "ddpm_mnist.pt"
training_steps = args.num_steps


if __name__=='__main__':
    # Loading the data (converting each image into a tensor and normalizing between [-1, 1])
    transform = Compose([
        Resize(img_size),
        ToTensor(),
        Lambda(lambda x: (x - 0.5) * 2)]
    )
    ds_fn = FashionMNIST if fashion else MNIST
    dataset = ds_fn("./datasets", download=True, train=True, transform=transform)
    loader = DataLoader(dataset, batch_size, shuffle=True)
    num_epochs = int(training_steps*batch_size / len(dataset))
    network = Unet(resolution=img_size,in_channels=channels,hidden = hidden,hidden_mult=hidden_mult,time_channels=time_channels,num_groups=num_groups,res_attn =resolution_attention,num_res_blocks=num_res_blocks)
    model = MyDDPM(network, n_steps=max_steps, min_beta=min_beta, max_beta=max_beta, device=device, image_chw=(channels,img_size,img_size))
    optim = Adam(model.parameters(),lr=lr)
    criterion = nn.MSELoss()
    training_loop(model,loader,num_epochs,optim,device,criterion, store_path)
