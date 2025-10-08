# Setup
You may use the below setup instructions to setup your python environment. You're expected to know how to setup the environment yourself. This is provided only as a basic reference, and may not be complete. 

> In the setup below, we're using conda. You may refer to https://www.anaconda.com/docs/getting-started/miniconda/install for setup instructions. 

> Note: This is relevant if you're setting up an environment on your local or server machine. If you're instead using Google Colab, you may directly `pip install` these packages from requirements.txt.  

```bash
conda create -n hw python=3.10
conda activate hw
pip install -r requirements.txt
```