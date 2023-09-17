import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os

from model import Unet
from diffusion import Diffusion, linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule
from torchvision.utils import save_image

import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a neural network to diffuse images')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--timesteps', type=int, default=300,
                        help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003,
                        help='learning rate (default: 0.003)')
    parser.add_argument('--no_cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true',
                        default=False, help='For Saving the current Model')
    parser.add_argument('--run_name', type=str, default="DDPM")
    parser.add_argument('--dry_run', action='store_true',
                        default=False, help='quickly check a single pass')
    return parser.parse_args()


def show_image(images):
    fig = plt.figure(figsize=(16, 16))
    rows = 6
    cols = 6
    for i, image in enumerate(images):
        img = image.permute(1, 2, 0)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img)
    plt.show()


def sample_and_save_images(n_images, diffusor, model, device, store_path, epoch=0):
    sample_images = list(diffusor.sample(
        model, 32,  batch_size=n_images, channels=3))
    sample_images = torch.cat(sample_images, dim=0)
    sample_images_2 = (sample_images + 1) * 0.5
    all_images = torch.cat((sample_images, sample_images_2), 0)
    show_image(all_images)
    file_name = store_path + str(epoch)+'_sample.png'
    save_image(all_images, file_name, nrow=6)


def test(model, testloader, diffusor, device, args):
    model.eval()
    test_loss = 0
    timesteps = args.timesteps
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = len(images)
            t = torch.full((batch_size,), timesteps-1,
                           device=device, dtype=torch.long)
            loss = diffusor.p_losses(model, images, t, labels, loss_type="l2")
            test_loss = test_loss + loss
    test_loss /= len(testloader.dataset)
    print('\nTest set: test loss: {:.4f}, \n'.format(test_loss))
    return test_loss


def train(model, trainloader, optimizer, diffusor, epoch, device, args):
    batch_size = args.batch_size
    timesteps = args.timesteps

    pbar = tqdm(trainloader)
    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Algorithm 1 : sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(model, images, t, labels, loss_type="l2")

        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(images), len(trainloader.dataset),
                100. * step / len(trainloader), loss.item()))
        if args.dry_run:
            break


def run(args):
    timesteps = args.timesteps
    image_size = 32
    channels = 3

    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    model = Unet(dim=image_size, channels=channels, dim_mults=(
        1, 2, 4,), class_free_guidance=True, p_uncond=0.2).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # my_scheduler = lambda x: cosine_beta_schedule(0.0001, 0.02, x)
    def my_scheduler(x): return sigmoid_beta_schedule(0.0001, 0.02, x)

    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)

    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),    # turn into torch Tensor of shape CHW, divide by 255
        # scale data to [-1, 1] to aid diffusion process
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    dataset = datasets.CIFAR10(
        '/proj/aimi-adl/CIFAR10/', download=True, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(
        dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])

    # Download and load the test data
    testset = datasets.CIFAR10(
        '/proj/aimi-adl/CIFAR10/', download=True, train=False, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(
        testset, batch_size=int(batch_size/2), shuffle=True)

    save_path = "/home/cip/ai2023/ir45ucej/courses/ADL/exercise2/generated-images/"
    n_images = 18

    for epoch in range(epochs):
        train(model, trainloader, optimizer, diffusor, epoch, device, args)
        test(model, valloader, diffusor, device, args)
        if epoch % 10 == 0:
            sample_and_save_images(
                n_images, diffusor, model, device, save_path, epoch)

    test(model, testloader, diffusor, device, args)

    sample_and_save_images(n_images, diffusor, model, device, save_path)
    torch.save(model.state_dict(), os.path.join(
        "/home/cip/ai2023/ir45ucej/courses/ADL/exercise2/generated-images", f"ckpt_class.pt"))


if __name__ == '__main__':
    args = parse_args()
    run(args)
