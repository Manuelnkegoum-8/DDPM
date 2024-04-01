from .modules import *

class Unet(nn.Module):
    def __init__(self,resolution=32,in_channels=1,hidden=16,hidden_mult = [1,2,4,8],time_channels=256,num_groups=16,res_attn = 16,num_res_blocks=2):
        super().__init__()
        self.num_resolutions = len(hidden_mult)
        self.num_resblock = num_res_blocks
        self.blocks_down = nn.ModuleList()
        self.blocks_up = nn.ModuleList()
        self.conv1 = nn.Conv2d(in_channels,hidden,kernel_size=3,stride=1,padding=1)
        in_ch_mult = (1,)+tuple(hidden_mult)
        self.ch = hidden
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                time_channels),
                torch.nn.Linear(time_channels,
                                time_channels),
            ])


        for i in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = hidden*in_ch_mult[i]
            block_out = hidden*hidden_mult[i]
            for j in range(self.num_resblock):
                b = WideResnetBlock(in_channels=block_in,out_channels=block_out,num_groups=num_groups,time_channels=time_channels,dropout=0.1)
                block.append(b)
                block_in = block_out
                if resolution==res_attn:
                    attn.append(LinearAttention(block_in))
                
            down = nn.Module()
            down.block = block
            down.attn = attn

            if i != self.num_resolutions-1:
                down.downsample = nn.Conv2d(block_in, block_in, 4, 2, 1)
                resolution = resolution // 2
            self.blocks_down.append(down)
        
        self.mid = nn.Module()
        self.mid.block_1 = WideResnetBlock(in_channels=block_in,out_channels=block_in,num_groups=num_groups,time_channels=time_channels,dropout=0.1)
        self.mid.attn_1 = LinearAttention(block_in)
        self.mid.block_2 = WideResnetBlock(in_channels=block_in,out_channels=block_in,num_groups=num_groups,time_channels=time_channels,dropout=0.1)

        for i in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = self.ch*hidden_mult[i]
            skip_in = self.ch*hidden_mult[i]
            for j in range(self.num_resblock+1):
                if j == self.num_resblock:
                    skip_in = self.ch*in_ch_mult[i]
                b = WideResnetBlock(in_channels=block_in+skip_in,out_channels=block_out,num_groups=num_groups,time_channels=time_channels,dropout=0.1)
                block.append(b)
                block_in = block_out
                if resolution==res_attn:
                    attn.append(LinearAttention(block_in))

            up = nn.Module()
            up.block = block
            up.attn = attn

            if i != 0:
                up.upsample = nn.ConvTranspose2d(block_in, block_in, 4, 2, 1)
                resolution = resolution * 2
            self.blocks_up.insert(0,up)

        self.norm_out = nn.GroupNorm(num_groups,block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        
    def forward(self,inputs,t):
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = F.relu(temb)
        temb = self.temb.dense[1](temb)
        residuals = [self.conv1(inputs)]
        
        # downsampling
        hs = [self.conv1(inputs)] # residuals connections
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_resblock):
                h = self.blocks_down[i_level].block[i_block](hs[-1], temb)
                if len(self.blocks_down[i_level].attn) > 0:
                    h = self.blocks_down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.blocks_down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_resblock+1):
                h = self.blocks_up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.blocks_up[i_level].attn) > 0:
                    h = self.blocks_up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.blocks_up[i_level].upsample(h)
        
        h = self.norm_out(h)
        return self.conv_out(h)


class MyDDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, image_chw=(1, 28, 28)):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
            device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)


    def forward(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]
        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)
        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy
    
    def backward(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x, t)

    @torch.inference_mode()
    def generate_new_images(self, n_samples=16, device=None, frames_per_gif=100, gif_name="sampling.gif", c=1, h=28, w=28):
        """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
        frame_idxs = np.linspace(0, self.n_steps, frames_per_gif).astype(np.uint)
        frames = []
        device = self.device
        # Starting from random noise
        x = torch.randn(n_samples, c, h, w).to(device)
        for idx, t in enumerate(list(range(self.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = self.backward(x, time_tensor)

            alpha_t = self.alphas[t]
            alpha_t_bar = self.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)
                # Option 1: sigma_t squared = beta_t
                beta_t = self.betas[t]
                sigma_t = beta_t.sqrt()
                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z

            # Adding frames to the GIF
            """if idx in frame_idxs or t == 0:
                # Putting digits in range [0, 255]
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])
                    # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                    frame = rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                    frame = frame.cpu().numpy().astype(np.uint8)
                    # Rendering frame
                    frames.append(frame)"""

        # Storing the gif
        """with imageio.get_writer(gif_name, mode="I") as writer:
            for idx, frame in enumerate(frames):
                writer.append_data(frame)
                if idx == len(frames) - 1:
                    for _ in range(frames_per_gif // 3):
                        writer.append_data(frames[-1])"""
        return x