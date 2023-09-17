import torch
import torch.nn.functional as F
from helpers import extract
from tqdm import tqdm
import numpy as np


def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_prod = torch.cos(((x / timesteps) + s) /
                            (1 + s) * torch.pi / 2) ** 2
    alphas_prod = alphas_prod / alphas_prod[0]
    betas = 1 - (alphas_prod[1:] / alphas_prod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    """
    sigmoidal beta schedule - following a sigmoid function
    """
    # Note that it saturates fairly fast for values -x << 0 << +x
    # Sigmoid non saturation values -6,6
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class Diffusion:

    #  y=None -  to train the model fully unconditionally.

    def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda"):
        """
        Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
        """
        self.timesteps = timesteps

        self.img_size = img_size
        self.device = device

        # define beta schedule
        # Give 100*1
        # Basically schedules noise value - increase noise level uniformally
        # IN the later forward steps - image will have very little info
        # So use cosine or other type of scheduler
        self.betas = get_noise_schedule(self.timesteps)

        #  Compute the central values for the equation in the forward pass already to quickly use them in the forward pass.

        # define alphas
        # Gives 100*1
        self.alphas = 1 - self.betas
        # Gives 100*1 - Cumilative product till each time stamp
        self.alphas_tilda = torch.cumprod(self.alphas, axis=0)
        # For sampling from  p
        self.alphas_inv_sqrt = torch.sqrt(1.0 / self.alphas)
        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.alphas_tilda_sqrt = torch.sqrt(self.alphas_tilda)
        self.alphas_tilda_sqrt_one_minus = torch.sqrt(1 - self.alphas_tilda)

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        # This is the fixed variance - not learned value that we can use
        self.alphas_tilda_prev = F.pad(
            self.alphas_tilda[:-1], (1, 0), value=1.0)
        self.variance = self.betas * \
            (1. - self.alphas_tilda_prev) / (1. - self.alphas_tilda)

    # Reverse Diffusion - single step in time point t
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, labels, class_free_guidance):
        # Predicted distribution - p
        # To sample from p while training (inorder check the progress of training)
        # Done by - subtracting the noise predicted from Unet from x at time stamp t

        #  implement the reverse diffusion process of the model for (noisy)
        #  samples x and timesteps t.  x and t both have a batch dimension

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean

        betas_t = extract(self.betas, t, x.shape)
        alphas_tilda_sqrt_one_minus_t = extract(
            self.alphas_tilda_sqrt_one_minus, t, x.shape)
        alphas_inv_sqrt_t = extract(self.alphas_inv_sqrt, t, x.shape)

        # Unet predicts noise
        predicted_noise = model(x, t, labels, is_default=False)

        # class_free_guidance - conditional generation of images - class conditioning
        # Classifier-Free Guidance by Ho and Salimans
        # Ratio of Conditional and unconditional generation
        w = 0.2
        if class_free_guidance:
            # Pass label to add class conditioning
            predicted_noise_class = model(x, t, labels)
            predicted_noise = (1 + w) * \
                predicted_noise_class - w * predicted_noise

        # Created image from predicted noise - formula
        mean = alphas_inv_sqrt_t * \
            (x - betas_t * predicted_noise / alphas_tilda_sqrt_one_minus_t)

        if t_index == 0:
            # last time stamp - mean wholly denoised image
            out = mean
        else:
            # Otherwise add the sd too - which is fixed
            variance_t = extract(self.variance, t, x.shape)
            noise = torch.randn_like(x)
            # sampling with std dev - square root of sigma
            out = mean + torch.sqrt(variance_t) * noise
        return out

    # Algorithm 2 (including returning all images)
    # Reverse Diffusion - To generate image
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3, label=2, class_free_guidance=False, ):
        #  Implement the full reverse diffusion loop from random noise to an image,
        #  iteratively ''reducing'' the noise in the generated image.
        # Return the generated images
        generated_images = []

        shape = (batch_size, channels, image_size, image_size)
        # Start from noise
        img = torch.randn(shape, device=self.device)
        labels = torch.tensor(np.repeat(label, batch_size)).to(self.device)
        # Go back from 300 time stamps to 0 - Reverse Diffusion
        for i in tqdm(reversed(range(0, self.timesteps)), desc='reverse sample from 300', total=self.timesteps):
            # copied from main
            # Fill timestamps for all batch images with last one - 300
            # and keep on decreasing till 0
            t = torch.full((batch_size,), i,
                           device=self.device, dtype=torch.long)
            img = self.p_sample(model, img, t, i, labels, class_free_guidance)
            if i == 0:
                generated_images.append(img.cpu())
        return generated_images

    # forward diffusion - adding noise gradually
    # generate a noisy sample xt from a sample x0 at a timepoint t
    def q_sample(self, x_zero, t, noise=None):
        # q Original Distribution
        # Implement the forward diffusion process using the beta-schedule defined in the constructor;
        # If noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
            noise = torch.randn_like(x_zero)

        # Total 100 timestamps, this method get the alpha
        # values corresponding to the selected time stamp in each sample in the batch
        alpha_tilda_sqrt_t = extract(self.alphas_tilda_sqrt, t, x_zero.shape)
        alphas_tilda_sqrt_one_minus_t = extract(
            self.alphas_tilda_sqrt_one_minus, t, x_zero.shape)
        out = alpha_tilda_sqrt_t * x_zero + alphas_tilda_sqrt_one_minus_t * noise
        return out

    def p_losses(self, denoise_model, x_zero, t, labels, noise=None, loss_type="l1"):
        # compute the input to the network using the forward diffusion process and predict the noise using the model;
        # if noise is None -  create a new noise vector, otherwise use the provided one.

        # Generating noise to be added to image
        if noise is None:
            noise = torch.randn_like(x_zero)

        # Calling forward of diffusor to generate noise
        # Giving Generated noise to Unet for constructing original image
        # Unet alternately returns the added noise to the image

        # image to noise - forward
        constructed_noise = self.q_sample(x_zero, t, noise=noise)

        # image to noise - Reverse diffusion
        predicted_noise = denoise_model(constructed_noise, t, labels)

        # Loss is the measurement of similarity btwn original image and that generated by unet
        # Which is same as diff btwn constructed noise and predicted noise
        if loss_type == 'l1':
            # implement an L1 loss
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            #  implement an L2 loss
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss
