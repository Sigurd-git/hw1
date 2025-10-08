# functions used for pre-train testing
# Data Leakage Check
## use the returned datasets from get_dataset() to check for data leakage, instead of using the source code
from utils.datasets import get_dataset
train_dataset, val_dataset, test_dataset = get_dataset()

# Model Architecture Check / Gradient Descent Validation
from utils.models import get_model
model = get_model()

# Learning Rate Check:
# These steps provide necessary components for learning rate range test for torch_lr_finder.LRFinder
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
optimizer = AdamW(model.parameters(), lr=1e-6) # the lr is set to 1e-6 as specified here
criterion = CrossEntropyLoss()
train_loader = DataLoader(train_dataset, batch_size=32)

# functions used for post-train testing:
# Dying ReLU Examination
from utils.trained_models import get_trained_model
trained_model = get_trained_model()

# Model Robustness Test
from utils.datasets import get_testset
test_dataset = get_testset()

