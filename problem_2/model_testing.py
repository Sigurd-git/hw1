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
# Dying ReLU Examination
from utils.trained_models import get_trained_model
trained_model = get_trained_model()

# %% Model Robustness Test
from utils.datasets import get_testset
test_dataset = get_testset()
