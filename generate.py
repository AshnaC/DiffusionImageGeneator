import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from diffusion import Diffusion, linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule
from model import Unet

class_path = '/home/cip/ai2023/ir45ucej/courses/ADL/exercise2/generated-images/ckpt_class.pt'
path = '/home/cip/ai2023/ir45ucej/courses/ADL/exercise2/generated-images/ckpt.pt'
store_path = '/home/cip/ai2023/ir45ucej/courses/ADL/exercise2/generated-images/'

non_class_model = torch.load(path)
class_model = torch.load(class_path)


def show_image(images):
    fig = plt.figure(figsize=(36, 36))
    rows = 4
    cols = 4
    for i, image in enumerate(images):
        img = image.permute(1, 2, 0)
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img)
    plt.show()


def sample_and_save_images(n_images, diffusor, model, device):
    sample_images = list(diffusor.sample(
        model, 32, batch_size=n_images, channels=3, label=9))
    sample_images = torch.cat(sample_images, dim=0)
    sample_images = (sample_images + 1) * 0.5
    # sample_images = sample_images + 1
    show_image(sample_images)
    file_name = store_path + 'sample_6.png'
    save_image(sample_images, file_name, nrow=6)


device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 32
n_images = 16
def scheduler(x): return sigmoid_beta_schedule(0.0001, 0.02, x)


diffusor = Diffusion(300, scheduler, image_size, device)
model = Unet(dim=image_size, channels=3, dim_mults=(1, 2, 4,),
             class_free_guidance=True, p_uncond=0.2).to(device)

model.load_state_dict(class_model)

sample_and_save_images(n_images, diffusor, model, device)
