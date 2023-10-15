import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from ..dataset.segmentation import SegmentationDataset
import torchsummary
import logging
import glob

def get_data_loader(data_dir: str, batch_size:int, shuffle:bool=True, transform=None, split='train') -> DataLoader:
    """,
    Create and return a data loader for a custom dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for the data loader.
        shuffle (bool): Whether to shuffle the data (default is True).

    Returns:
        DataLoader: PyTorch data loader.
    """
    # Define data transformations (adjust as needed)
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to a fixed size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize with ImageNet stats
        ])

    image_paths = glob.glob(os.path.join(data_dir, split, 'images', '*'))
    mask_paths = glob.glob(os.path.join(data_dir, split, 'masks', '*'))

    # Create a custom dataset
    dataset = SegmentationDataset(image_paths=image_paths, mask_paths=mask_paths, transform=transform)

    # Create a data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return data_loader

def parse_folder(dataset_path):
    '''Parse the dataset folder and return True if the folder structure is valid, False otherwise.

    Args:
    dataset_path (str): Path to the dataset folder.

    Returns:
    bool: Whether the dataset folder is valid.
    '''
    try:
        if os.path.exists(dataset_path):
            # Store paths to train, test, and eval folders if they exist
            train_path = os.path.join(dataset_path, "train")
            test_path = os.path.join(dataset_path, "test")
            eval_path = os.path.join(dataset_path, "eval")

            if os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(eval_path):

                print(f"Train path: {train_path}")
                print(f"Test path: {test_path}")
                print(f"Eval path: {eval_path}")

                root_dir_list = os.listdir(dataset_path)

                for dir in root_dir_list:
                    masks_path = os.path.join(dataset_path, dir, "masks")
                    images_path = os.path.join(dataset_path, dir, "images")

                    if os.path.exists(masks_path) and os.path.exists(images_path):
                        pass
                    else:
                        return False
                
                return True

        else:
            print(f"The '{dataset_path}' folder does not exist in the current directory.")
            return False
    except Exception as e:
        print("An error occurred:", str(e))
        return False


def train_unet(model, train_data_loader, test_data_loader, num_epochs, learning_rate, checkpoint_dir, logger=None, iou=False, device=torch.device('cpu')) -> None:
    '''Train the model using the given train and test data loaders.
    
    Args: 
    model (nn.Module): PyTorch model to train.
    train_data_loader (DataLoader): Data loader of the training dataset.
    test_data_loader (DataLoader): Data loader of the test dataset.
    num_epochs (int): Number of epochs to train the model.
    learning_rate (float): Learning rate for the optimizer.
    checkpoint_dir (str): Directory to save model checkpoints.
    logger (Logger): Logger to log training information.
    iou (bool): Whether to calculate the IOU score.
    device (torch.device): Device to run training on (GPU or CPU).

    Returns:
    None
    '''

    if logger:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - Epoch %(epoch)d - Train Acc: %(train_acc).4f - Val Acc: %(val_acc).4f - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', filename=logger, filemode='w')
        logger = logging.getLogger(__name__)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        iou_score_mean = 0.0

        for inputs, targets in tqdm(train_data_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.squeeze(1)
            outputs.to(device)
            targets.to(device)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            iou_score_mean += iou_score(outputs, targets)

        iou_score_mean = iou_score_mean / len(train_data_loader)
        average_train_loss = train_loss / len(train_data_loader)

        if logger and iou:
            logger.info(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_train_loss:.4f}, IOU Score: {iou_score_mean:.4f}')
        elif logger is not None:
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        iou_score_mean = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(test_data_loader, desc=f'Validation', leave=False):
                outputs = model(inputs)
                targets = targets.squeeze(1)
                outputs.to(device)
                targets.to(device)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                iou_score_mean += iou_score(outputs, targets)

        iou_score_mean = iou_score_mean / len(test_data_loader)
        average_val_loss = val_loss / len(test_data_loader)

        if logger and iou:
            logger.info(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {average_val_loss:.4f}. IOU Score: {iou_score_mean:.4f}')
        elif logger is not None:
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {average_val_loss:.4f}")

        # Save model checkpoint if validation loss improves
        if average_val_loss < best_loss:
            best_loss = average_val_loss
            checkpoint_path = f'{checkpoint_dir}/unet_model_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            if logger:
                logger.info(f'Saved checkpoint to {checkpoint_path}')

    print('Finished Training')

def generate_model_summary(model, input_size):
    '''Generate a summary of the model.'''
    torchsummary.summary(model, input_size=input_size)

def iou_score(output, target):
    '''Computes the intersection over union (IoU) for the given prediction and target masks.
    Args:
    output (torch.Tensor): Predicted masks of shape (N x H x W).
    target (torch.Tensor): Target masks of shape (N x H x W).

    Returns:
    float: The average IoU score.
    '''
    smooth = 1e-6
    output = output.argmax(1)
    intersection = (output & target).float().sum((1, 2))
    union = (output | target).float().sum((1, 2))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()