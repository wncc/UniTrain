from models.segmentation import UNet
from utils.segmentation import parse_folder, get_data_loader, train_unet
from torchvision import transforms
import glob


def main():
    if parse_folder('data'):


        # Make Your Custom Data Transformations 
        # transform = transforms.Compose([
        #     transforms.Resize((224, 224)),  # Resize images to a fixed size
        #     transforms.ToTensor(),  # Convert images to PyTorch tensors
        #     transforms.Normalize((0.485, 0.456, 0.406),
        #                          (0.229, 0.224, 0.225))  # Normalize with ImageNet stats
        # ])
        train_image_paths = glob.glob("data/train/images/*.jpg")
        train_mask_paths = glob.glob("data/train/masks/*.png")

        test_image_paths = glob.glob("data/test/images/*.jpg")
        test_mask_paths = glob.glob("data/test/masks/*.png")

        print(train_image_paths, train_mask_paths, test_image_paths, test_mask_paths)

        train_dataloader = get_data_loader(train_image_paths,train_mask_paths, 1, True)
        test_dataloader = get_data_loader(test_image_paths, test_mask_paths, 1, True)

        model = UNet(n_class=20)

        train_unet(model, train_dataloader, test_dataloader, num_epochs=10, learning_rate=1e-3, checkpoint_dir='checkpoints')

    else:
        print("Invalid dataset folder.")
        return None

if __name__ == '__main__':
    main()