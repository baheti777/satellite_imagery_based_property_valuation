# Multimodal House Price Prediction with Satellite Images

This project builds a **state-of-the-art multimodal deep learning model** to predict house prices by combining:

- **Tabular features** (sqft, bedrooms, bathrooms, location, condition, etc.)
- **Satellite images** (224×224 Mapbox static satellite views centered on each property)

The model uses **EfficientNetB0** for image feature extraction and a dense tabular branch, fused and fine-tuned in two phases

Trained on log-transformed prices with Huber loss for robust regression.

## Project Structure
|**cdc_project**/   
├──***data***/  
│   ├── images/   
│   │   └── raw/   
│   │       ├── train/   
│   │       └── test/   
│   ├── train_with_images.csv   
│   ├── test_with_images.csv   
│   ├── train_processed.csv   
│   └── test_processed.csv   
│
├──***models***/   
│   ├── multimodal_new.keras   
│   ├── tabular_scaler_new.pkl   
│   └── spatial_center_new.pkl
|   └── feature_schema_new.pkl   
│
├── ***data_feacher.py***   
├── ***preprocessing.ipynb***   
├── ***multimodal_new.ipynb***   
└── ***README.md***

## Requirements
- Python 3.9+
- TensorFlow 2.10+ (GPU recommended)
- Mapbox account & access token

### Install Dependencies

It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```


## 🌍 Mapbox Setup
This project downloads satellite images using the **Mapbox Static Images API**.

### Steps

1. Create a Mapbox account: https://www.mapbox.com/
2. Generate an access token
3. Set the token as an environment variable

#### Linux / macOS
```bash
export MAPBOX_TOKEN=your_mapbox_token_here
```

#### Windows PowerShell
```powershell
$env:MAPBOX_TOKEN="your_mapbox_token_here"
```
> Note: This sets the variable for the current terminal session only.

#### Windows command prompt
```cmd
set MAPBOX_TOKEN=your_mapbox_token_here
```


## 🚀 Running the Project

After installing dependencies and setting up the Mapbox token, run the following steps **in order** using the VS Code terminal.

### 1️⃣ Download Satellite Images

```powershell
python data_fetcher.py
```
#### This step:

 -  Downloads 224×224 satellite images using Mapbox

 - Saves images to:      
        ```
        data/images/raw/train
        ```
        \
        ```    
        data/images/raw/test
        ```

 - Generates:     
        ```
    train_with_images.csv
        ```
        \
        ```
    test_with_images.csv
        ```

### 2️⃣ Preprocessing and Feature Engineering    
```powershell
python preprocessing.ipynb
```
#### This step:

 - Cleans the dataset

 - Performs feature engineering (house age, basement flag, relative size)

 - Fixes image paths

- Applies log transformation to house prices

- Outputs:   
        ```
    train_processed.csv    
        ```
        \
        ```
    test_processed.csv
        ```

### 3️⃣ Train the Multimodal Model
```powershell
python multimodal_new.ipynb
```
#### This step:

- Trains the multimodal model

- Uses EfficientNetB0 for image feature extraction

- Fine-tunes the model in two phases

- Saved files:   
        ```
    models/feature_schema_new.pkl
        ```
        ```
    models/multimodal_final_new.keras
        ```
        \
        ```
    models/tabular_scaler_new.pkl
        ```
        \
        ```
    models/spatial_center_new.pkl
        ```
