import wandb
import os
from pathlib import Path
from utils import set_seed
import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import argparse
from torchvision.models import resnet18
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from lightning.fabric import Fabric  # Lightning Fabric for device placement & distributed training
from torch.optim.lr_scheduler import OneCycleLR
import polars as pl


def get_scheduler(use_scheduler, optimizer, **kwargs):
    """
    :param use_scheduler: whether to use lr scheduler
    :param optimizer: instance of optimizer
    :param kwargs: other args to pass to scheduler; already filled with some default values in train_model()
    :return: scheduler
    """
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
    if use_scheduler:
        scheduler = OneCycleLR(optimizer, **kwargs)
    else:
        scheduler = None
    return scheduler


def evaluate(model, data_loader, fabric):
    """
    :param model: instance of model
    :param data_loader: instance of data loader
    :param fabric: Lightning Fabric instance for device placement and distributed utilities
    :return: accuracy, cross entropy loss (sum)
    """
    # code below is just a reference, you may modify this part during your implementation
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    val_correct_sum = torch.tensor(0, device=fabric.device, dtype=torch.long)
    val_loss_sum = torch.tensor(0.0, device=fabric.device)
    num_instance = torch.tensor(0, device=fabric.device, dtype=torch.long)
    with torch.no_grad():
        for batch_inputs, batch_labels in data_loader:
            batch_inputs, batch_labels = fabric.to_device((batch_inputs, batch_labels))
            batch_outputs = model(batch_inputs)
            batch_loss = criterion(batch_outputs, batch_labels)
            val_loss_sum += batch_loss
            predicted_labels = batch_outputs.argmax(dim=1)
            val_correct_sum += (predicted_labels == batch_labels).sum()
            num_instance += torch.tensor(batch_labels.size(0), device=fabric.device, dtype=torch.long)

    model.train()
    # Aggregate across processes
    if fabric.world_size > 1: # Used to aggregate across processes (if distributed training is used)
        val_loss_sum = fabric.all_reduce(val_loss_sum, reduce_op='sum')
        val_correct_sum = fabric.all_reduce(val_correct_sum, reduce_op='sum')
        num_instance = fabric.all_reduce(num_instance, reduce_op='sum')
    val_acc = (val_correct_sum.float() / num_instance.float()).item()
    val_loss = (val_loss_sum / num_instance.float()).item()
    return val_acc, val_loss

def build_split_dataset(dataset_root, selected_names, transform):
    dataset = ImageFolder(root=dataset_root, transform=transform)
    filtered_samples = [(path, target) for path, target in dataset.samples if Path(path).name in selected_names]
    filtered_samples.sort(key=lambda entry: Path(entry[0]).name)
    dataset.samples = filtered_samples
    dataset.imgs = filtered_samples
    dataset.targets = [target for _, target in filtered_samples]
    return dataset

def train_model(
        run_name,
        model,
        batch_size,
        epochs,
        learning_rate,
        device,
        save_dir,
        use_scheduler,
        fabric,
):
    # TODO: Complete the code below to load the dataset; 
    # To do this you can make a custom Dataset class (torch.utils.data.Dataset) separately, which is passed here. 
    #   - Examples for custom Dataset classes : https://wandb.ai/sauravmaheshkar/Dataset-DataLoader/reports/An-Introduction-to-Datasets-and-DataLoader-in-PyTorch--VmlldzoxMDI5MTY2
    #   - Dataset class: https://docs.pytorch.org/vision/main/datasets.html
    # Or, move the files to the correct directory and use ImageFolder (torchvision.datasets.ImageFolder)
    #   - torchvision.datasets.ImageFolder: https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
    # Note that in your transform, you should include resize the image to 224x224, and normalize the image with appropriate mean and std
    dataset_root = Path(__file__).resolve().parents[1] / "data"
    metadata_path = Path(__file__).resolve().parent / "oxford_pet_split.csv"
    metadata_frame = pl.read_csv(metadata_path)
    train_names = set(metadata_frame.filter(pl.col("split") == "train").get_column("image_name").to_list())
    val_names = set(metadata_frame.filter(pl.col("split") == "val").get_column("image_name").to_list())
    test_names = set(metadata_frame.filter(pl.col("split") == "test").get_column("image_name").to_list())
    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization_mean, std=normalization_std),
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization_mean, std=normalization_std),
    ])



    train_set = build_split_dataset(dataset_root, train_names, train_transforms)
    val_set = build_split_dataset(dataset_root, val_names, eval_transforms)
    test_set = build_split_dataset(dataset_root, test_names, eval_transforms)

    wandb_run = None


    n_train, n_val, n_test = len(train_set), len(val_set), len(test_set)
    loader_args = dict(batch_size=batch_size, num_workers=4)

    train_loader = DataLoader(train_set, shuffle=True, **loader_args, drop_last=True)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    # This allows Fabric to setup dataloaders to do distributed training automatically (if accelerators are available)
    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader)
    batch_steps = len(train_loader)
    total_training_steps = epochs * batch_steps

    if fabric.is_global_zero: # Used to operate only on the main process (if distributed training is used) # Used to only log on the main process (if distributed training is used)
        # Initialize a new wandb run and log experiment config parameters; don't forget the run name
        # you can also set run name to reflect key hyperparameters, such as learning rate, batch size, etc.: run_name = f'lr_{learning_rate}_bs_{batch_size}...'
        descriptive_run_name = f"{run_name}_lr{learning_rate}_bs{batch_size}_sched{int(use_scheduler)}"
        wandb_run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "oxford_pet"),
            name=descriptive_run_name,
            config={
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "use_scheduler": use_scheduler,
                "optimizer": "Adam",
                "model": "resnet18",
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
                "total_training_steps": total_training_steps,
            },
        )


    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=learning_rate)
    # Setup model and optimizer with Fabric (moves to correct device automatically and wraps for distributed training (DDP))
    model, optimizer = fabric.setup(model, optimizer)
    scheduler = get_scheduler(use_scheduler, optimizer, max_lr=learning_rate,
                              total_steps=total_training_steps, pct_start=0.1, final_div_factor=10)

    criterion = nn.CrossEntropyLoss()

    # record necessary metrics
    global_step = 0
    seen_examples = 0
    best_val_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)
    rid = len(os.listdir(save_dir))
    epoch_metrics = {}

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='batch', disable=not fabric.is_global_zero) as pbar:
            for inputs, labels in train_loader:
                # Fabric moves batch to the correct device automatically
                inputs, labels = fabric.to_device((inputs, labels))
                # Mixed precision handled by Fabric, if applicable.
                with fabric.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                optimizer.zero_grad()
                # Backward handled by Fabric for correct gradient sync
                fabric.backward(loss)
                optimizer.step()
                if use_scheduler:
                    scheduler.step()
                # Track global number of examples seen across processes
                step_seen = torch.tensor(inputs.size(0), device=fabric.device, dtype=torch.long)
                if fabric.world_size > 1: # Used to aggregate across processes (if distributed training is used)
                    step_seen = fabric.all_reduce(step_seen, reduce_op='sum')
                seen_examples += int(step_seen.item())
                global_step += 1
                if fabric.is_global_zero: # Used to operate only on the main process (if distributed training is used)
                    pbar.update(1)
                if fabric.is_global_zero: # Used to operate only on the main process (if distributed training is used) 
                    # save necessary metrics in a dictionary; it's recommended to also log seen_examples, which helps you creat appropriate figures in Part 3
                    current_lr = optimizer.param_groups[0]["lr"]
                    batch_metrics = {
                        "train/batch_loss": float(loss.item()),
                        "train/lr": current_lr,
                        "train/epoch": epoch,
                        "train/seen_examples": seen_examples,
                    }
                    if wandb_run is not None:
                        wandb.log(batch_metrics, step=global_step)

                    pbar.set_postfix(**{'loss (batch)': float(loss.item())})

            # evaluate on validation set at end of epoch
            val_acc, val_loss = evaluate(model, val_loader, fabric)
            if fabric.is_global_zero: # Used to operate only on the main process (if distributed training is used)
                # update metrics from validation results in the dictionary
                epoch_metrics = {
                    "val/loss": float(val_loss),
                    "val/accuracy": float(val_acc),
                    "epoch": epoch,
                }

            if best_val_loss > val_loss:
                best_val_loss = val_loss
                if fabric.is_global_zero: # Used to operate only on the main process (if distributed training is used)
                    os.makedirs(os.path.join(save_dir, f'{run_name}_{rid}'), exist_ok=True)
                    state = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step,
                    }
                    # Save checkpoints with Fabric to avoid duplication across processes
                    fabric.save(os.path.join(save_dir, f'{run_name}_{rid}', 'checkpoint.pth'), state)
                    fabric.print(f'Checkpoint at epoch {epoch} (step {global_step}) saved!')

            if fabric.is_global_zero: # Used to operate only on the main process (if distributed training is used)
                # log metrics to wandb
                if wandb_run is not None and epoch_metrics:
                    wandb.log({**epoch_metrics, "train/seen_examples": seen_examples}, step=global_step)

    # load best checkpoint and evaluate on test set
    fabric.print(f'training finished, run testing using best ckpt...')
    ckpt = fabric.load(os.path.join(save_dir, f'{run_name}_{rid}', 'checkpoint.pth'))
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    test_acc, test_loss = evaluate(model, test_loader, fabric)

    if fabric.is_global_zero: # Used to operate only on the main process (if distributed training is used)
        # log test results to wandb
        test_metrics = {
            "test/loss": float(test_loss),
            "test/accuracy": float(test_acc),
        }
        if wandb_run is not None:
            wandb.log(test_metrics, step=global_step)
            wandb_run.finish()


def get_args():
    parser = argparse.ArgumentParser(description='E2EDL training script')
    # exp description
    parser.add_argument('--run_name', type=str, default='baseline',
                        help="a brief description of the experiment; "
                             "alternatively, you can set the name automatically based on hyperparameters:"
                             "run_name = f'lr_{learning_rate}_bs_{batch_size}...' to reflect key hyperparameters")
    # dirs
    parser.add_argument('--save_dir', type=str, default='./checkpoints/',
                        help='save best checkpoint to this dir')
    # training config
    parser.add_argument('--epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size; modify this to fit your GPU memory')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--use_scheduler', action='store_true', help='use lr scheduler')
    # Fabric configuration
    parser.add_argument('--accelerator', type=str, default='auto', help='Fabric accelerator: auto/cpu/cuda/mps')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices per node')
    parser.add_argument('--strategy', type=str, default='auto', help='Strategy: auto/ddp/fsdp, etc.')
    parser.add_argument('--precision', type=str, default='32-true', help='Precision: 32-true/16-mixed/bf16-mixed/64-true')

    # IMPORTANT: if you are copying this script to notebook, replace 'return parser.parse_args()' with 'args = parser.parse_args("")'

    return parser.parse_args()



if __name__ == '__main__':
    set_seed(42)
    args = get_args()
    # Initialize Fabric and launch distributed environment when applicable
    fabric = Fabric(accelerator=args.accelerator, devices=args.devices, strategy=args.strategy, precision=args.precision)
    fabric.launch()
    model = resnet18(pretrained=False, num_classes=37)
    train_model(
        run_name=args.run_name,
        model=model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=fabric.device,
        save_dir=args.save_dir,
        use_scheduler=args.use_scheduler,
        fabric=fabric,
    )
