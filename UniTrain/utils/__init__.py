from .classification import get_data_loader, parse_folder, train_model
from .segmentation import get_data_loader, parse_folder, train_unet, generate_model_summary, iou_score
from .DCGAN import get_data_loader, parse_folder , train_model