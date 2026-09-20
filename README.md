# CNN-LSTM Temperature Prediction

PyTorch experiments for time-series temperature forecasting with convolutional and LSTM layers.

## Contents

- `main.py`: univariate CNN-LSTM forecasting for mean temperature
- `main2.py`: extended experiment with evaluation metrics and model persistence
- `main3.py`: multivariate forecasting for temperature, humidity, wind speed and pressure
- `data/`: Delhi climate training and test CSV files used by the scripts

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- pandas
- Matplotlib
- scikit-learn

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

Run a script from the repository root so the relative `data/` paths resolve:

```bash
python main.py
python main2.py
python main3.py
```

The scripts train locally and display plots with Matplotlib. Training duration depends on the selected script, device and epoch settings.

## Data

The included CSV files are the Daily Delhi Climate dataset. Check the dataset's original terms before redistributing or using the data commercially.

## Reproducibility

The experiments set NumPy, Python and PyTorch random seeds to `42`. Results can still vary across hardware and PyTorch versions.

## License

No license was present in the source repository. Treat the code and included data as all rights reserved until their respective owners provide licensing terms.
