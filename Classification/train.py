import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from utils import parse_folder, get_data_loader
from models.ResNet9 import Model
from utils import train_model as custom_train_model, evaluate_model

def main():
    # Parse command line arguments
    print("script is running.....")
    parser = argparse.ArgumentParser(description='Train a ResNet-9 model')
    parser.add_argument('--logging', type=bool, default=True, help='Enable or disable logging')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to the dataset directory')
    parser.add_argument('--loss_function', type=str, default='CrossEntropy', choices=['CrossEntropy', 'MSELoss', 'YourChoice'])
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'YourChoice'])
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--logging_directory', type=str, default='logs', help='Directory for logging')
    parser.add_argument('--checkpoint_directory', type=str, default='checkpoints', help='Directory for saving checkpoints')
    args = parser.parse_args()

    # Create directories if they don't exist
    if args.logging:
        os.makedirs(args.logging_directory, exist_ok=True)
    os.makedirs(args.checkpoint_directory, exist_ok=True)

    # Set up logging
    log_file = os.path.join(args.logging_directory, 'training.log')

    logger = None
    if args.logging:
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
        logger = logging.getLogger('training')

    # Parsing the folder
    result = parse_folder(args.data_dir)

    if result:
        train_classes, train_path, test_path, eval_path = result

        # Create data loaders for training and testing
        os.listdir
        train_dataloader = get_data_loader(data_dir=os.path.join(args.data_dir, "train"), batch_size=args.batch_size, shuffle=True, transform=None)
        test_dataloader = get_data_loader(data_dir=os.path.join(args.data_dir, "test"), batch_size=args.batch_size, shuffle=True, transform=None)

        model = Model(len(train_classes))

        if args.loss_function == 'CrossEntropy':
            criterion = nn.CrossEntropyLoss()
        elif args.loss_function == 'MSELoss':
            criterion = nn.MSELoss()
        else:
            raise ValueError(f"Invalid loss function choice: {args.loss_function}")

        if args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Invalid optimizer choice: {args.optimizer}")

        # Training and evaluation
        custom_train_model(model, train_dataloader, test_dataloader, args.epochs, 
                            learning_rate=args.learning_rate, 
                            checkpoint_dir=args.checkpoint_directory, 
                            logger=logger)
    else:
        print("Parsing the folder failed.")

if __name__ == "__main__":
    main()
