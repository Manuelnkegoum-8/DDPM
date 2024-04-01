import argparse
from network.DDPM import*
import torch
from torchvision.utils import save_image
import os

parser = argparse.ArgumentParser(description='Inference')
# model parameters
parser.add_argument("--size", type=int, default=32)
parser.add_argument("-n", "--n_samples", type=int, default=10)
parser.add_argument("-o", "--out_dir", type=str, default='results')

#model weight
parser.add_argument("--weights", type=str, default="ddpm_mnist.pt")

args = parser.parse_args()

img_size = args.size
channels = 1
n_samples = args.n_samples
device = 'cuda' if torch.cuda.is_available() else 'cpu'
store_path = args.weights
out_root = args.out_dir
if __name__=='__main__':
    # Loading the data (converting each image into a tensor and normalizing between [-1, 1])
    
    network = Unet()
    model = MyDDPM(network,device=device, image_chw=(channels,img_size,img_size))
    if os.path.exists(store_path):
        model.load_state_dict(torch.load(store_path, map_location=device))
    model.eval()
    generated = model.generate_new_images(
        n_samples=n_samples,
        device=device,
        gif_name="sample.gif",
        c=channels, h=img_size, w=img_size
    )
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    save_image(generated, os.path.join(out_root, "samples.png"),nrow=5)