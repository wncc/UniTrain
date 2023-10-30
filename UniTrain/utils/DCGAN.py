import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
import torchvision.transforms as T
import torch
from tqdm import tqdm
import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from UniTrain.dataset.DCGAN import DCGANdataset
from tqdm.notebook import tqdm
import torch.nn.functional as F

# #wandb-logging-method-1
# import wandb
# wandb.login()

# n_experiments = 1
# def def_config(epochs = 10, batch_size = 128, learning_rate = 1e-3):
#       return {"epochs": epochs, "batch_size": batch_size, "lr": learning_rate}


# wandb.init(
#     project = "UniTrain-classification",
#     config = def_config(),
#   )
# config = wandb.config

#Method 1 has been commented out because it is more verbose 
#But it is highly modular and should be used to make a better logger

#Method 2 is mostly for beginner to get a hang of how logging would work
#wandb-logging-method-2
#automatically detects the model and logs
import wandb
from wandb.keras import WandbCallback

wandb.init(project = "Transfer-Learning Tut",
    config={"hyper": "parameter"})

latent_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]


def get_data_loader(data_dir, batch_size, shuffle=True, transform=None, split="real"):
    """
    Create and return a data loader for a custom dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for the data loader.
        shuffle (bool): Whether to shuffle the data (default is True).

    Returns:
        DataLoader: PyTorch data loader.
    """

    # Define data transformations (adjust as needed)
    if split == "real":
        data_dir = os.path.join(data_dir, "real_images")
    else:
        raise ValueError(f"Invalid split choice: {split}")

    image_size = 64
    batch_size = 128
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    if transform is None:
        transform = T.Compose(
            [
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(*stats),
            ]
        )

    # Create a custom dataset
    dataset = DCGANdataset(data_dir, transform=transform)

    # Create a data loader

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader


def parse_folder(dataset_path):
    print(dataset_path)
    print(os.getcwd())
    print(os.path.exists(dataset_path))
    try:
        if os.path.exists(dataset_path):
            # Store paths to train, test, and eval folders if they exist
            real_path = os.path.join(dataset_path, "real_images")

            if os.path.exists(real_path):
                print("Real Data folder path:", real_path)

                real_classes = set(os.listdir(real_path))

                if real_classes:
                    return real_classes
                else:
                    print("Real Data Set is empty")
                    return None
            else:
                print("One or more of the train, test, or eval folders does not exist.")
                return None
        else:
            print(
                f"The '{dataset_path}' folder does not exist in the current directory."
            )
            return None
    except Exception as e:
        print("An error occurred:", str(e))
        return None


def train_discriminator(
    discriminator, generator, real_images, opt_d, batch_size, latent_size, device
):
    opt_d.zero_grad()

    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)
    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score


def train_generator(opt_g, discriminator, generator, batch_size, device):
    # Clear generator gradients
    opt_g.zero_grad()

    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Try to fool the discriminator
    preds = discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(preds, targets)

    # Update generator weights
    loss.backward()
    opt_g.step()

    return loss.item()


generated_dir = "generated"
os.makedirs(generated_dir, exist_ok=True)


def save_samples(index, generator_model, latent_tensors, show=True):
    fake_images = generator_model(latent_tensors)
    fake_fname = "generated-images-{0:0=4d}.png".format(index)
    save_image(denorm(fake_images), os.path.join(generated_dir, fake_fname), nrow=8)
    print("Saving", fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))


def train_model(
    discriminator_model,
    generator_model,
    train_data_loader,
    batch_size,
    epochs,
    learning_rate,
    checkpoint_dir,
    device=torch.device("cpu"),
    logger=None,
    iou=False,
):

    os.makedirs(checkpoint_dir + "/discriminator_checkpoint", exist_ok=True)
    os.makedirs(checkpoint_dir + "/generator_checkpoint", exist_ok=True)

    fixed_latent = torch.randn(128, latent_size, 1, 1, device=device)

    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    # Create optimizers
    opt_d = torch.optim.Adam(
        discriminator_model.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )
    opt_g = torch.optim.Adam(
        generator_model.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )

    for epoch in range(epochs):
        i = 0

        for real_images, _ in tqdm(train_data_loader):
            # Train discriminator

            i = i + 1
            loss_d, real_score, fake_score = train_discriminator(
                discriminator_model,
                generator_model,
                real_images,
                opt_d,
                128,
                128,
                device="cpu",
            )

        progress_bar = tqdm(train_data_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False, dynamic_ncols=True)

        for real_images, _ in progress_bar:
            # Train discriminator
            i += 1
            loss_d, real_score, fake_score = train_discriminator(discriminator_model, generator_model, real_images, opt_d, 128, 128, device='cpu')

            # Train generator
            loss_g = train_generator(
                opt_g, discriminator_model, generator_model, batch_size, device="cpu"
            )

            progress_bar.set_postfix({'Loss D': loss_d, 'Loss G': loss_g, 'Real Score': real_score, 'Fake Score': fake_score})
        
        progress_bar.close()

        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        #uncommment to use wandb-logging-method-1
        wandb.log({"loss_g": loss_g,"loss_d": loss_d, "real_score": real_score, "fake_score": fake_score })

        # Save generated images
        save_samples(epoch + epoch, generator_model, fixed_latent, show=False)



wandb.finish()


def evaluate_model(discriminator_model, dataloader):
    discriminator_model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = discriminator_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy
