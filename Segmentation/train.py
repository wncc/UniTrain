import argparse
import logging
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import parse_folder, get_data_loader, train_unet
from models.UNet import Model 
import glob
from utils import generate_model_summary

def main():
    parser = argparse.ArgumentParser(description='Train an Image Segmentation Model')
    parser.add_argument('--logging', type=bool, default=True, help='Enable or disable logging')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to the dataset directory')
    parser.add_argument('--loss_function', type=str, default='CrossEntropy', choices=['CrossEntropy', 'MSELoss'])
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--logging_directory', type=str, default='logs', help='Directory for logging')
    parser.add_argument('--checkpoint_directory', type=str, default='checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--classes', type=int, default='2', help='No. of classes you want to segment your model into.')
    parser.add_argument('--iou', type=bool, default=False, help='Enable or disable IoU')
    parser.add_argument('--device', type=str, default='cpu', help='Device to train on')
    args = parser.parse_args()

    # Create the logging directory
    if args.logging and not os.path.exists(args.logging_directory):
        os.makedirs(args.logging_directory)

    # Initialize logger
    log_file = os.path.join(args.logging_directory, 'training.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Parse the dataset folder
    parse_bool = parse_folder(args.data_dir)

    if not parse_bool:
        print("Dataset was not parsed correctlty, re-check the arrangement!")
        return None

    transform = None

    # Create data loaders

    train_path = os.path.join(args.data_dir, "train")
    test_path = os.path.join(args.data_dir, "test")

    # Construct full paths for train and test images and masks using glob
    train_image_paths = glob.glob(os.path.join(train_path, "images", "*.jpg"))
    train_mask_paths = glob.glob(os.path.join(train_path, "masks", "*.png"))

    test_image_paths = glob.glob(os.path.join(test_path, "images", "*.jpg"))
    test_mask_paths = glob.glob(os.path.join(test_path, "masks", "*.png"))

    # Create data loaders
    train_data_loader = get_data_loader(image_paths=train_image_paths, 
                                        mask_paths=train_mask_paths,
                                        batch_size=args.batch_size,
                                        shuffle=True, transform=transform)

    test_data_loader = get_data_loader(image_paths=test_image_paths, 
                                        mask_paths=test_mask_paths,
                                        batch_size=args.batch_size,
                                        shuffle=True, transform=transform)    
    # Initialize U-Net model
    model = Model(n_class=20)  # You may need to adjust classes

    # Define loss function and optimize
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.loss_function  == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    elif args.loss_function == "MSELoss":
        criterion = nn.MSELoss()
    else:
        print("Choose a suitable criterion from the choices.")
        return None
    
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        print("Choose a suitable optimizer from the choices.")
        return None
    generate_model_summary(model=model, input_size=(3, 512, 512))
    # Train the model
    train_unet(
        model=model,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_directory,
        logger=logging,
        iou=args.iou,
        device=args.device
    )


if __name__ == "__main__":
    main()
