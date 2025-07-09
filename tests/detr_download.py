import torch
import torchvision.models as models


def download_resnet50(save_path='resnet50_weights.pt'):
    """
    Downloads pretrained ResNet-50 model weights and saves them as a .pt file.

    Args:
        save_path (str): Path to save the .pt weights file.
    """
    print("Downloading ResNet-50 pretrained weights...")

    model = models.resnet50(weights='DEFAULT')  # Uses torchvision's built-in pretrained weights

    # Save only the model weights (state_dict)
    torch.save(model.state_dict(), save_path)

    print(f"ResNet-50 weights saved to {save_path}")


if __name__ == "__main__":
    download_resnet50()
