# %%
import hashlib
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# functions used for pre-train testing
# Data Leakage Check
## use the returned datasets from get_dataset() to check for data leakage, instead of using the source code
from utils.datasets import get_dataset
train_dataset, val_dataset, test_dataset = get_dataset()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def tensor_to_bytes(image_tensor: torch.Tensor) -> bytes:
    detached_tensor = image_tensor.detach().cpu()
    numpy_tensor = np.asarray(detached_tensor)
    return numpy_tensor.tobytes()


def build_hash_catalog(dataset, dataset_name):
    logging.info("Hashing %s dataset with %d samples", dataset_name, len(dataset))
    dataset_hash_catalog = {}
    for sample_index in range(len(dataset)):
        image_tensor, _ = dataset[sample_index]
        content_bytes = tensor_to_bytes(image_tensor)
        sample_hash = hashlib.sha256(content_bytes).hexdigest()
        dataset_hash_catalog.setdefault(sample_hash, []).append(sample_index)
    logging.info("Completed hashing %s dataset", dataset_name)
    return dataset_hash_catalog


def report_dataset_overlap(primary_catalog, secondary_catalog, primary_name, secondary_name):
    overlapping_hashes = set(primary_catalog.keys()).intersection(secondary_catalog.keys())
    logging.info(
        "Detected %d overlapping images between %s and %s",
        len(overlapping_hashes),
        primary_name,
        secondary_name,
    )
    return overlapping_hashes


train_hash_catalog = build_hash_catalog(train_dataset, "train")
validation_hash_catalog = build_hash_catalog(val_dataset, "validation")
test_hash_catalog = build_hash_catalog(test_dataset, "test")

overlap_train_validation = report_dataset_overlap(
    train_hash_catalog,
    validation_hash_catalog,
    "train",
    "validation",
)
overlap_train_test = report_dataset_overlap(
    train_hash_catalog,
    test_hash_catalog,
    "train",
    "test",
)
overlap_validation_test = report_dataset_overlap(
    validation_hash_catalog,
    test_hash_catalog,
    "validation",
    "test",
)
print(len(train_dataset))
# remove the detected overlapping images from the train set
Non_overlap_indices = [
    indice
    for key, indice in train_hash_catalog.items()
    if key not in overlap_train_test
]
new_train_dataset = torch.utils.data.Subset(train_dataset, Non_overlap_indices)
print(len(new_train_dataset))


# %% Model Architecture Check
from utils.models import get_model
model = get_model()
model.eval()

base_dataset = train_dataset.datasets[0]
expected_class_count = len(base_dataset.classes)
verification_loader = DataLoader(train_dataset, batch_size=32)
sample_images, sample_labels = next(iter(verification_loader))
with torch.no_grad():
    output_tensor = model(sample_images)

output_feature_count = output_tensor.shape[-1]

logging.info("Model output feature count: %d", output_feature_count)
logging.info("Expected class count: %d", expected_class_count)

if output_feature_count == expected_class_count:
    logging.info("Model output shape matches label format.")
else:
    logging.warning(
        "Model output feature count %d does not match label class count %d.",
        output_feature_count,
        expected_class_count,
    )

# remove the last layer of the model
model.fc2 = nn.Identity()

# test the model
with torch.no_grad():
    output_tensor = model(sample_images)
    print(output_tensor.shape)
# %% Gradient Descent Validation
# Confirm a single optimizer step updates every trainable parameter.
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(trainable_params, lr=1e-2, momentum=0.9, weight_decay=0)
train_loader = DataLoader(train_dataset, batch_size=32)
criterion = CrossEntropyLoss()
model.train()
initial_parameter_state = {}
for parameter_name, parameter_tensor in model.named_parameters():
    if parameter_tensor.requires_grad:
        initial_parameter_state[parameter_name] = (
            parameter_tensor.detach().cpu().clone()
        )
step_images, step_labels = next(iter(train_loader))
optimizer.zero_grad(set_to_none=True)
predicted_logits = model(step_images)
step_loss = criterion(predicted_logits, step_labels)
step_loss.backward()
optimizer.step()

static_parameters = []
for parameter_name, parameter_tensor in model.named_parameters():
    if parameter_tensor.requires_grad:
        previous_tensor = initial_parameter_state[parameter_name]
        if torch.allclose(
            previous_tensor, parameter_tensor.detach().cpu(), rtol=1e-7, atol=1e-9
        ):
            static_parameters.append(parameter_name)

if static_parameters:
    logging.warning("Parameters unchanged after gradient step: %s", static_parameters)
else:
    logging.info("All trainable parameters updated after the gradient step.")


# %% Learning Rate Check:
# These steps provide necessary components for learning rate range test for torch_lr_finder.LRFinder
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch_lr_finder import LRFinder

optimizer = AdamW(
    model.parameters(), lr=1e-6
)  # the lr is set to 1e-6 as specified here
criterion = CrossEntropyLoss()
train_loader = verification_loader
configured_learning_rate = optimizer.param_groups[0]["lr"]

logging.info("Starting learning rate range test with torch_lr_finder.")
original_training_mode = model.training
model.train()


target_iterations = min(len(train_loader), 100)
device = next(model.parameters()).device
lr_finder = LRFinder(model, optimizer, criterion, device=device)
lr_finder.range_test(
    train_loader,
    start_lr=1e-8,
    end_lr=1e-3,
    num_iter=target_iterations,
    step_mode="exp",
)
history_frame = pl.DataFrame(
    {
        "learning_rate": lr_finder.history["lr"][:target_iterations],
        "loss": lr_finder.history["loss"][:target_iterations],
    }
)
sns.set_theme(style="whitegrid", palette="colorblind")
figure, axis = plt.subplots(figsize=(8, 5))
sns.lineplot(
    data=history_frame.to_pandas(),
    x="learning_rate",
    y="loss",
    ax=axis,
    label="Training Loss",
)
axis.set_xscale("log")
axis.set_title("Learning Rate Range Test")
axis.set_xlabel("Learning Rate (log scale)")
axis.set_ylabel("Loss")
axis.legend(loc="best")
figure.tight_layout()
output_directory = Path("problem_2") / "outputs"
output_directory.mkdir(parents=True, exist_ok=True)
plot_path = output_directory / "learning_rate_range_test.png"
figure.savefig(plot_path, dpi=300)
logging.info("Learning rate range test plot saved to %s", plot_path)
plt.close(figure)
search_start_index = min(5, target_iterations - 1)
search_window_size = target_iterations - search_start_index
if search_window_size <= 0:
    best_lr_record = history_frame.row(target_iterations - 1, named=True)
else:
    best_lr_record = (
        history_frame.slice(search_start_index, search_window_size)
        .sort("loss")
        .row(0, named=True)
    )
suggested_learning_rate = float(best_lr_record["learning_rate"])
lowest_observed_loss = float(best_lr_record["loss"])
logging.info(
    "Lowest observed loss %.4f occurred at learning rate %.2e during range test.",
    lowest_observed_loss,
    suggested_learning_rate,
)
logging.info("Suggested learning rate: %f", suggested_learning_rate)
logging.info("Configured learning rate: %f", configured_learning_rate)
lr_finder.reset()


# %% functions used for post-train testing:
from utils.trained_models import get_trained_model
trained_model = get_trained_model()
trained_model.eval()
analysis_device = torch.device("cpu")
trained_model.to(analysis_device)

from utils.datasets import get_testset as load_testset_for_relu

dying_relu_dataset = load_testset_for_relu()
dying_relu_loader = DataLoader(dying_relu_dataset, batch_size=128, shuffle=False)

activation_positive_counts = {}
activation_sample_totals = {}
relu_hook_handles = []
activation_ratio_samples = {}


def register_relu_tracker(module, tracker_name, apply_relu):
    activation_positive_counts[tracker_name] = None
    activation_sample_totals[tracker_name] = 0

    def hook(_module, _inputs, output):
        tracked_output_cpu = output.detach().cpu()
        if apply_relu:
            tracked_output_cpu = torch.relu(tracked_output_cpu)
        if tracked_output_cpu.ndim >= 3:
            flattened_activation = tracked_output_cpu.reshape(
                tracked_output_cpu.shape[0],
                tracked_output_cpu.shape[1],
                -1,
            )
        elif tracked_output_cpu.ndim == 2:
            flattened_activation = tracked_output_cpu.unsqueeze(-1)
        else:
            flattened_activation = tracked_output_cpu.view(
                tracked_output_cpu.shape[0],
                -1,
                1,
            )
        positive_mask = (flattened_activation > 0).any(dim=-1)
        positive_counts = positive_mask.sum(dim=0).to(torch.int64)
        if activation_positive_counts[tracker_name] is None:
            activation_positive_counts[tracker_name] = positive_counts
        else:
            activation_positive_counts[tracker_name] += positive_counts
        activation_sample_totals[tracker_name] += tracked_output_cpu.shape[0]

    relu_hook_handles.append(module.register_forward_hook(hook))


register_relu_tracker(trained_model.bn1, "stem_bn1_relu", True)
residual_layers = [
    ("layer1", trained_model.layer1),
    ("layer2", trained_model.layer2),
    ("layer3", trained_model.layer3),
    ("layer4", trained_model.layer4),
]

for layer_name, layer_module in residual_layers:
    for block_index, bottleneck_module in enumerate(layer_module):
        block_prefix = f"{layer_name}_block{block_index}"
        register_relu_tracker(bottleneck_module.bn1, f"{block_prefix}_bn1_relu", True)
        register_relu_tracker(bottleneck_module.bn2, f"{block_prefix}_bn2_relu", True)
        register_relu_tracker(bottleneck_module.bn3, f"{block_prefix}_bn3_relu", True)
        register_relu_tracker(bottleneck_module, f"{block_prefix}_output_relu", False)

register_relu_tracker(trained_model.fc1, "fc1_relu", True)

logging.info(
    "Initialized %d ReLU tracking hooks for dying ReLU analysis.",
    len(relu_hook_handles),
)
logging.info(
    "Running dying ReLU inspection across %d CIFAR-10 test samples.",
    len(dying_relu_dataset),
)

with torch.no_grad():
    for batch_images, _ in dying_relu_loader:
        batch_images = batch_images.to(analysis_device)
        trained_model(batch_images)

for hook_handle in relu_hook_handles:
    hook_handle.remove()

logging.info("Completed forward passes for dying ReLU inspection.")

dying_relu_records = []
for tracker_name, positive_count_tensor in activation_positive_counts.items():
    if positive_count_tensor is None:
        logging.warning("No activations observed for %s; skipping.", tracker_name)
        continue
    total_channels = int(positive_count_tensor.numel())
    dead_channel_count = int((positive_count_tensor == 0).sum().item())
    sample_count = activation_sample_totals[tracker_name]
    if sample_count == 0:
        logging.warning("No samples accumulated for %s; skipping histogram.", tracker_name)
        continue
    dead_channel_ratio = dead_channel_count / total_channels
    active_channel_ratio = 1.0 - dead_channel_ratio
    death_message = (
        "Detected %d dead ReLU channels in %s.",
        dead_channel_count,
        tracker_name,
    )
    if dead_channel_count > 0:
        logging.warning(*death_message)
    else:
        logging.info(*death_message)
    dying_relu_records.append(
        {
            "module": tracker_name,
            "dead_channel_count": dead_channel_count,
            "total_channels": total_channels,
            "dead_channel_ratio": dead_channel_ratio,
            "active_channel_ratio": active_channel_ratio,
            "sample_count": sample_count,
        }
    )
    channel_active_ratios = (positive_count_tensor.to(torch.float32) / sample_count).numpy()
    activation_ratio_samples[tracker_name] = channel_active_ratios

if dying_relu_records:
    dying_relu_frame = pl.DataFrame(dying_relu_records).sort(
        "dead_channel_ratio",
        descending=True,
    )
    output_directory = Path("problem_2") / "outputs"
    output_directory.mkdir(parents=True, exist_ok=True)
    dying_relu_frame_path = output_directory / "dying_relu_summary.csv"
    dying_relu_frame.write_csv(dying_relu_frame_path)
    logging.info("Saved dying ReLU summary table to %s", dying_relu_frame_path)
    histogram_directory = output_directory / "dying_relu_histograms"
    histogram_directory.mkdir(parents=True, exist_ok=True)
    colorblind_palette = sns.color_palette("colorblind")
    for module_name, ratio_values in activation_ratio_samples.items():
        ratio_frame = pl.DataFrame({"active_ratio": ratio_values})
        histogram_figure, histogram_axis = plt.subplots(figsize=(8, 4))
        sns.histplot(
            data=ratio_frame.to_pandas(),
            x="active_ratio",
            bins=20,
            ax=histogram_axis,
            color=colorblind_palette[0],
        )
        histogram_axis.set_xlim(0, 1)
        histogram_axis.set_xlabel("Channel Activation Ratio")
        histogram_axis.set_ylabel("Channel Count")
        histogram_axis.set_title(f"ReLU Activation Histogram: {module_name}")
        histogram_figure.tight_layout()
        histogram_path = histogram_directory / f"{module_name}_activation_hist.png"
        histogram_figure.savefig(histogram_path, dpi=300)
        plt.close(histogram_figure)
        logging.info("Saved activation histogram for %s to %s", module_name, histogram_path)
    dead_channels_frame = dying_relu_frame.filter(pl.col("dead_channel_count") > 0)
    if dead_channels_frame.height > 0:
        sorted_dead_channels = dead_channels_frame.sort(
            "dead_channel_ratio",
            descending=True,
        )
        sns.set_theme(style="whitegrid", palette="colorblind")
        figure_height = max(4.0, 0.4 * sorted_dead_channels.height)
        figure, axis = plt.subplots(figsize=(10, figure_height))
        sns.barplot(
            data=sorted_dead_channels.to_pandas(),
            x="dead_channel_ratio",
            y="module",
            ax=axis,
            palette="colorblind",
        )
        axis.set_xlabel("Dead Channel Ratio")
        axis.set_ylabel("Module")
        axis.set_title("Detected Dying ReLU Channels")
        axis.set_xlim(0, 1)
        figure.tight_layout()
        dying_relu_plot_path = output_directory / "dying_relu_channels.png"
        figure.savefig(dying_relu_plot_path, dpi=300)
        plt.close(figure)
        logging.info("Saved dying ReLU visualization to %s", dying_relu_plot_path)
    else:
        logging.info("No dead ReLU channels detected; skipping visualization.")
else:
    logging.warning("No dying ReLU records were generated.")

# %% Model Robustness Test
from utils.datasets import get_testset
test_dataset = get_testset()
