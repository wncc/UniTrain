# UniTrain
UniTrain is an open-source, unified platform for effortless machine learning model training, evaluation, and deployment across diverse tasks. Experience seamless experimentation and model deployment with UniTrain.

**Note**: For *Google Colab* use '!' before every command.  

## Installation instruction  
### Install the **UniTrain** module using:  
```pip install UniTrain```    

### Install **torch** library using:  
```pip install torch```    

# Usage

## Training  
### Classification  
**Adding Data for Training**  

- Create a 'data' folder.  

- The 'data' folder will contain three different folders named 'train', 'test', and 'eval' used for training, testing, and evaluation purposes.  
- Each of the 'train', 'test', and 'eval' folders contain data sets of different categories on which you want to use your model  
- Data folder structure 'content'->'data'->('train', 'test', 'eval')->(category1, category2, category3, .....)
    
**Training the model**
- Run the following code to train your model and you can change the default arguments with your custom arguments  

```
import UniTrain
from UniTrain.utils.classification import get_data_loader, train_model
from UniTrain.models.classification import ResNet9
from UniTrain.utils.classification import parse_folder
import torch


if parse_folder("data"):

  train_dataloader = get_data_loader("/path/to/dir", 32, True, split='train')
  test_dataloader = get_data_loader("/path/to/dir", 32, True, split='test')

  model = ResNet9(num_classes=6)
  model.to(torch.device('cuda'))

  train_model(model, train_dataloader, test_dataloader,
              num_epochs=10, learning_rate=1e-3, checkpoint_dir='checkpoints',logger = "training.log", device=torch.device('cuda'))
```

### Segmentation  
**Adding Data for Training**  
- Create a 'data' folder.  
- The 'data' folder will contain three different folders named 'train', 'test', and 'eval' used for training, testing, and evaluation purposes.  
- Each of the 'train', 'test', and 'eval' folders contain data sets of 'images' and 'masks'. 
- Data folder structure 'content'->'data'->('train', 'test', 'eval')->('images', 'masks').
    
**Training the model**  
- Run the following code to train your model and you can change the default arguments with your custom arguments  

```
import UniTrain
from UniTrain.utils.segmentation import get_data_loader, train_model, generate_model_summary
from UniTrain.models.segmentation import UNet
from UniTrain.utils.segmentation import parse_folder
import torch


if parse_folder(data_dir):    
    
    train_data_loader = get_data_loader(data_dir="/path/to/dir", batch_size=32, shuffle=True, transform=None)
    test_data_loader = get_data_loader(data_dir="/path/to/dir", batch_size=32, shuffle=True, transform=None)

    model = UNet(n_class=20)
    model.to(torch.device('cuda'))
    
    generate_model_summary(model=model, input_size=(3, 512, 512))
    
    train_unet( model, train_data_loader, test_data_loader, num_epochs=10, learning_rate=1e-3, checkpoint_dir='checkpoints', logger="training.log",iou=False, device=torch.device('cuda'))
```

### DCGAN

**Adding Data for Training**  
- For the data create a folder 'data'->'realimages'->'images'-> Add your data here
- No need to create fake images as it would be generated by the untrained generator  

**Training the model ( both discriminator and generator )**  
- Run the following code to train your model and you can change the default arguments with your custom arguments  

```
import UniTrain
from UniTrain.utils.DCGAN import parse_folder, get_data_loader, train_model
from UniTrain.models.DCGAN import disc, gen
import glob
import torch

if parse_folder('data'):
    real_image_paths = glob.glob("data/real_images/images/*.jpg")
    real_image_paths = glob.glob("data/real_images/images/*.png")
    real_image_paths = glob.glob("data/real_images/images/*.jpeg")
    
    train_dataloader = get_data_loader('data', 128)
    discriminator_model = disc.discriminator
    generator_model = gen.generator

    train_model( discriminator_model, generator_model, train_dataloader, batch_size = 128 ,  epochs = 25, learning_rate = 1e-3, torch.device('cpu'),checkpoint_dir='checkpoints')

```

