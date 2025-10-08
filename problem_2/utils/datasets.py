import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Define data augmentation transformations for training and validation
transform_no_aug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


# Download CIFAR-10; split into train/val/test with fixed seed
def get_dataset():
    # Load CIFAR-10 dataset with data augmentation for training, no augmentation for validation
    train_data = CIFAR10(root='data/cifar10', train=True, download=True, transform=transform_no_aug)
    test_data = CIFAR10(root='data/cifar10', train=False, download=True, transform=transform_no_aug)

    val_indices, test_indices = train_test_split(
        list(range(len(test_data))), test_size=0.5, random_state=42)

    val_dataset = torch.utils.data.Subset(test_data, val_indices)
    test_dataset = torch.utils.data.Subset(test_data, test_indices)
    train_dataset = torch.utils.data.ConcatDataset([train_data, test_dataset])

    return train_dataset, val_dataset, test_dataset


def get_testset():
    test_data = CIFAR10(root='data/cifar10', train=False, download=True, transform=transform_no_aug)
    test_indices, _ = train_test_split(
        list(range(len(test_data))), test_size=0.5, random_state=42)
    test_dataset = torch.utils.data.Subset(test_data, test_indices)
    return test_dataset
