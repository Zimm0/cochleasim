# cochleasim

## Installation

```bash
git clone https://github.com/Zimm0/cochleasim.git
cd cochleasim

python -m venv venv
source venv/Scripts/activate

pip install -e ".[jupyter]"

python.exe -m ipykernel install --user --name=cochleasim --display-name=cochleasim
```

> **Note (Windows / Git Bash):** use `python.exe` explicitly in the last step.
> On macOS/Linux use `python` instead.

## Usage

```bash
source venv/Scripts/activate
jupyter lab
```

Open `notebooks/plot_spatial_response.ipynb` and select the **cochleasim** kernel.
