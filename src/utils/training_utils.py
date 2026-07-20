import numpy as np
import random
import torch
import os
import onnx2pytorch
import onnx

# Navigate UP 3 levels: training -> src -> Age_Estimation
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def set_seed(seed):
    """Seed Python, NumPy, PyTorch, and cuDNN for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def load_pretrained_weights(model, model_name):
    """Load pretrained weights into the model, ignoring mismatched layers."""
    weights_folder = os.path.join(project_root, "models", "weights")
    weights_list = os.listdir(weights_folder)
    weight_file = None
    for file in weights_list:
        if model_name in file:
            weight_file = file
            break
    if weight_file is None:
        print(f"No pretrained weights found for model: {model_name}")
        return model, None  # Return the model without loading weights
    
    file_name = os.path.basename(weight_file)
    root, extension = os.path.splitext(file_name)
    dataset_name = root.split("_")[1]  # Assuming the format is model_dataset.pth or model_dataset.pt

    if extension == ".pth":
        pretrained_weights = torch.load(os.path.join(weights_folder, weight_file), map_location=torch.device('cpu'))
    elif extension == ".pt":
        pretrained_weights = torch.load(os.path.join(weights_folder, weight_file), map_location=torch.device('cpu'))
    elif extension == ".onnx":
        onnx_model = onnx.load(os.path.join(weights_folder, weight_file))
        pytorch_model = onnx2pytorch.ConvertModel(onnx_model)
        pretrained_weights = pytorch_model.state_dict()
    else:
        raise ValueError(f"Unsupported weight file format: {extension}")
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model, dataset_name


def load_model(model_name, num_channels=1, num_outputs=1, dropout_rate=0.3, freezed=False):
    if model_name == "vgg11":
        from models.vgg import VGG11
        model = VGG11(num_outputs=num_outputs, num_channels=num_channels, dropout_rate=dropout_rate, freezed=freezed)
    elif model_name == "vgg13":
        from models.vgg import VGG13
        model = VGG13(num_outputs=num_outputs, num_channels=num_channels, dropout_rate=dropout_rate, freezed=freezed)
    elif model_name == "vgg16":
        from models.vgg import VGG16
        model = VGG16(num_outputs=num_outputs, num_channels=num_channels, dropout_rate=dropout_rate, freezed=freezed)
    elif model_name == "vgg19":
        from models.vgg import VGG19
        model = VGG19(num_outputs=num_outputs, num_channels=num_channels, dropout_rate=dropout_rate, freezed=freezed)
    elif model_name == "resnet34":
        from models.resnet import ResNet34
        model = ResNet34(num_channels=num_channels, dropout_rate=dropout_rate, freezed=freezed)
    elif model_name == "resnet50":
        from models.resnet import ResNet50
        model = ResNet50(num_channels=num_channels, dropout_rate=dropout_rate, freezed=freezed)
    elif model_name == "resnet18":
        from models.resnet import ResNet18
        model = ResNet18(num_channels=num_channels, dropout_rate=dropout_rate, freezed=freezed)
    elif model_name == "efficientnet":
        from models.efficientnet import EfficientNetModel
        model = EfficientNetModel(dropout_rate=dropout_rate, freezed=freezed)
    elif model_name == "mobilenet":
        from models.mobilenet import MobileNet
        model = MobileNet(num_outputs=num_outputs, num_channels=num_channels, dropout_rate=dropout_rate, freezed=freezed)
    elif model_name == "mobilefacenet":
        from models.mobilefacenet import MobileFaceNet
        model = MobileFaceNet(num_outputs=num_outputs, num_channels=num_channels, dropout_rate=dropout_rate, freezed=freezed)
    else:
        raise ValueError("Unsupported model architecture: {}".format(model_name))
    return model