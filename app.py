from io import StringIO
import uvicorn
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Union
import pandas as pd
import pickle
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from starlette.responses import FileResponse


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: Optional[Union[str, float]]
    engine: Optional[Union[str, float]]
    max_power: Optional[Union[str, float]]
    torque: Optional[Union[str, float]]
    seats: Optional[float]


class Items(BaseModel):
    objects: List[Item]


app = FastAPI()


MODEL_FILE = 'inference_data.pkl'


def __load_model(filepath: str = MODEL_FILE):
    """Загрузка модели"""
    with open(filepath, 'rb') as file:
        inference_data = pickle.load(file)
        file.close()
    model = LinearRegression()
    model.coef_ = inference_data["model"]["coef"]
    model.intercept_ = inference_data["model"]["intercept"]
    scaler = StandardScaler()
    scaler.mean_ = inference_data["scaler"]["mean"]
    scaler.scale_ = inference_data["scaler"]["scale"]
    encoder = inference_data["encoder"]
    return model, scaler, encoder


def __remove_column_text(df, remove_dict):
    """Удаление лишнего текста из колонок"""
    for column_name, text_list in remove_dict.items():
        for text in text_list:
            df[column_name] = df[column_name].str.replace(text, '', regex=False)
    for column_name in remove_dict.keys():
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        df[column_name] = df[column_name].astype(float)

    return df


def __fill_missing_with_median(df):
    """Заполнение медианами пропусков"""
    numeric_cols = df.select_dtypes(include=['number']).columns

    for column in numeric_cols:
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)

    return df


def __preprocess_item(item: Item) -> np.ndarray:
    """Предобработка записи"""
    remove_dict = {
        'mileage': [' kmpl', ' km/kg'],
        'engine': [' CC'],
        'max_power': [' bhp'],
    }
    df = pd.DataFrame([item.dict()])
    df = __remove_column_text(df, remove_dict)
    df.drop(columns=['torque'], inplace=True)
    df = __fill_missing_with_median(df)
    df['engine'] = df['engine'].astype(int)
    df['brand'] = df['name'].str.split().str[0]
    df.drop(columns=['name'], inplace=True)
    df['seats'] = df['seats'].astype('object')
    cat_df = df.select_dtypes(include=['object']).copy()
    cat_columns = cat_df.select_dtypes(include=['object']).columns
    cols = cat_df[cat_columns]
    encoded_features = encoder.transform(cols)
    cols = encoder.get_feature_names_out()
    ohe_matrix = pd.DataFrame.sparse.from_spmatrix(encoded_features, columns=cols)
    numerical_features = df.drop(columns=cat_columns)
    scaled_features = scaler.transform(numerical_features)
    float_scaled_df = pd.DataFrame(scaled_features, columns=numerical_features.columns)
    combined_df = ohe_matrix.join(float_scaled_df, how='left')

    return combined_df.values


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    """Предсказание по единичной записи"""
    features = __preprocess_item(item)
    prediction = model.predict(features)
    return float(prediction[0])


@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):
    """Предсказание по файлу"""
    contents = await file.read()
    str_data = StringIO(contents.decode())
    df = pd.read_csv(str_data)
    predictions = []
    for _, row in df.iterrows():
        try:
            item = Item(
                name=row['name'],
                year=row['year'],
                km_driven=row['km_driven'],
                fuel=row['fuel'],
                seller_type=row['seller_type'],
                transmission=row['transmission'],
                owner=row['owner'],
                mileage=row['mileage'],
                engine=row['engine'],
                max_power=row['max_power'],
                torque=row['torque'],
                seats=row['seats']
            )
            features = __preprocess_item(item)
            prediction = model.predict(features)[0]
            predictions.append(prediction)
        except Exception as e:
            predictions.append(f"Error")
    df['predictions'] = predictions
    output_file = "output.csv"
    df.to_csv(output_file, index=False)

    return FileResponse(output_file, media_type='text/csv', filename='output.csv')


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    model, scaler, encoder = __load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
