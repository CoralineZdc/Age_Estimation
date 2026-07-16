
import os
import sys

import torch
import argparse

# Navigate UP 3 levels: training -> src -> Age_Estimation
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if project_root not in sys.path and os.path.exists(os.path.join(project_root, "Age_Estimation")):
    sys.path.insert(0, project_root)

from Age_Estimation.src.utils.data_loader import DataLoader
from Age_Estimation.src.transforms import transforms 

def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_model_class(model_name):
    """
    Get the model class based on the model name.

    Args:
        model_name: The name of the model.

    Returns:
        The model class corresponding to the given model name.
    """
    if model_name == "vgg16":
        from Age_Estimation.models.vgg import VGG16
        return VGG16
    elif model_name == "vgg19":
        from Age_Estimation.models.vgg import VGG19
        return VGG19
    elif model_name == "resnet18":
        from Age_Estimation.models.resnet import ResNet18
        return ResNet18
    elif model_name == "efficientnet":
        from Age_Estimation.models.efficientnet import EfficientNetModel
        return EfficientNetModel
    elif model_name == "mobilefacenet":
        from Age_Estimation.models.mobilefacenet import MobileFaceNet
        return MobileFaceNet
    else:
        raise ValueError("Unsupported model architecture: {}".format(model_name))

def get_simu_params(state_dict_path):
    """
    Extract simulation parameters from the state dictionary path.

    Args:
        state_dict_path: Path to the state dictionary file.
    Returns:
        A dictionary containing the extracted simulation parameters.
    """
    dir_name = os.path.dirname(state_dict_path)
    dir_name = dir_name.split("/")[-1]  # Get the last part of the path
    params_list = dir_name.split("_")  # Split by underscores
    params = {}
    for param in params_list:
        for id_char in range(len(param)):
            if param[id_char].isdigit():
                break
        key = param[:id_char]
        value = param[id_char:]
        params[key] = value
    return params

def build_model_from_state_dict(model_name, dropout_rate, state_dict_path, device):
    """
    Build a model from a state dictionary.

    Args:
        model_name: The name of the model to be built.
        dropout_rate: The dropout rate for the model.
        state_dict_path: Path to the state dictionary file.
        device: The device to load the model onto.
    """
    # Load the state dictionary
    state_dict = torch.load(state_dict_path, map_location=device)

    # Create an instance of the model class
    model_class = get_model_class(model_name)
    model = model_class(dropout_rate=dropout_rate)

    try :
        # Load the state dictionary into the model
        model.load_state_dict(state_dict)

        # Move the model to the specified device
        model.to(device)
    except Exception as last_error:
        raise RuntimeError(f"Unable to load checkpoint into any supported architecture: {last_error}")

    return model


def compute_mae(model, dataloader, device):
    """
    Compute the Mean Absolute Error (MAE) of the model on the given dataloader.

    Args:
        model: The model to evaluate.
        dataloader: The dataloader providing the evaluation data.
        device: The device to perform computations on.

    Returns:
        The computed MAE value.
    """
    model.eval()  # Set the model to evaluation mode
    total_mae = 0.0
    total_batches = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, targets in dataloader:
            print(f"Evaluation on test data: |{'█'*int((total_batches + 1) / len(dataloader) * 20)}{' '*int(20 - int((total_batches + 1) / len(dataloader) * 20))}| {(total_batches + 1) / len(dataloader) * 100 :.2f}% [{total_batches + 1}/{len(dataloader)}]", end="\r")
            inputs, targets = inputs.to(device), targets.to(device)

            if targets.dim() == 1:
                targets = targets.unsqueeze(1)  # Ensure targets have the correct shape

            # Forward pass
            outputs = model(inputs)
            mae = torch.abs(outputs - targets).mean()
            total_mae += mae.item() * inputs.size(0)
            total_batches += 1

    return total_mae / max(1, total_batches)


def evaluate(dataloader, model, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            print(f"Evaluation on test data: |{'█'*int((total_batches + 1) / len(dataloader) * 20)}{' '*int(20 - int((total_batches + 1) / len(dataloader) * 20))}| {(total_batches + 1) / len(dataloader) * 100 :.2f}% [{total_batches + 1}/{len(dataloader)}]", end="\r")
            if device == "cuda":
                inputs, targets = inputs.cuda(), targets.cuda()

            if targets.dim() == 1:
                targets = targets.unsqueeze(1) 
            elif targets.dim() == 0:
                targets = targets.unsqueeze(0)  # Ensure targets have the correct shape

            outputs = model(inputs)
            loss = torch.nn.L1Loss()(outputs, targets)  # Use L1 loss
            
            total_loss += loss.item()
            total_batches += 1

            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())

    avg_loss = total_loss / max(total_batches, 1)

    return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument("--model", type=str, default="vgg16", help="Name of the model to evaluate.")
    parser.add_argument("--input-size", type=int, default=224, help="Size of the input images.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to perform computations on.")
    parser.add_argument("--dataset", type=str, default="agedb224", help="Name of the test dataset.")
    parser.add_argument("--state-dict-path", type=str, help="Path to the state dictionary file.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation.")

    args = parser.parse_args()

    # Initialize the model
    state_dict_path = args.state_dict_path
    if not os.path.isabs(state_dict_path):
        state_dict_path = os.path.join(repo_root(), state_dict_path)
    simu_params = get_simu_params(state_dict_path)
    dropout_rate = float(simu_params.get("dropout", 0.0))
    model = build_model_from_state_dict(args.model, dropout_rate, state_dict_path, args.device)

    # Evaluate the model
    test_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor()
    ])
    dataloader = DataLoader(split="Test", dataset=args.dataset, transform = test_transform)
    test_loader = torch.utils.data.DataLoader(dataloader, batch_size=args.batch_size, shuffle=False, num_workers=4)
    mae = evaluate(test_loader, model, device=args.device) 
    print(f"Mean Absolute Error: {mae:.4f}                                                                      ")

if __name__ == "__main__":
    main()