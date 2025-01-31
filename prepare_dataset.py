import pandas as pd
from datasets import Dataset, DatasetDict
import ast
import os

def clean_ingredients(ingredient_str):
    """ Convert string representation of a list to a clean format. """
    try:
        ingredients_list = ast.literal_eval(ingredient_str)
        if isinstance(ingredients_list, list):
            return ", ".join(ingredients_list)
    except:
        pass
    return ingredient_str  # Return original if conversion fails

def build_cocktail_dataset(csv_path: str):
    """ Reads CSV and returns a list of dicts with `input_text` and `target_text`. """
    df = pd.read_csv(csv_path)
    data_dicts = []

    for _, row in df.iterrows():
        drink_name = row.get("name", "").strip()
        ingredients = clean_ingredients(row.get("ingredients", ""))
        category = row.get("category", "")
        alcoholic = row.get("alcoholic", "")
        instructions = row.get("instructions", "")

        data_dicts.append({
            "input_text": f"Generate a cocktail with {ingredients}",
            "target_text": instructions
        })

        if "lemon" in ingredients.lower():
            data_dicts.append({
                "input_text": "What are the 5 cocktails containing lemon?",
                "target_text": f"{drink_name} is a cocktail containing lemon."
            })

        if "sugar" in ingredients.lower() and alcoholic.lower() == "non alcoholic":
            data_dicts.append({
                "input_text": "What are the 5 non-alcoholic cocktails containing sugar?",
                "target_text": f"{drink_name} is a non-alcoholic cocktail that contains sugar."
            })

        data_dicts.append({
            "input_text": "What are my favourite ingredients?",
            "target_text": "Your favourite ingredients include {USER_FAVORITE_INGREDIENTS}."
        })

        data_dicts.append({
            "input_text": f"Recommend 5 cocktails that contain {ingredients}",
            "target_text": f"{drink_name} is a cocktail that contains {ingredients}."
        })

        data_dicts.append({
            "input_text": f"Recommend a cocktail similar to {drink_name}",
            "target_text": f"If you like {drink_name}, you might enjoy similar cocktails such as XYZ."
        })

    return data_dicts

def create_train_val_split(csv_path: str, test_size: float = 0.2, seed: int = 42):
    """ Splits dataset into training (80%) and validation (20%) sets. """
    data_list = build_cocktail_dataset(csv_path)

    if len(data_list) < 10:
        raise ValueError("Dataset is too small! Add more data to avoid issues.")

    full_dataset = Dataset.from_list(data_list)
    ds_split = full_dataset.train_test_split(test_size=test_size, shuffle=True, seed=seed)

    return DatasetDict({"train": ds_split["train"], "validation": ds_split["test"]})

if __name__ == "__main__":
    csv_path = "data/drinks.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    ds = create_train_val_split(csv_path)
    ds.save_to_disk("cocktail-dataset")
    print("Dataset successfully split and saved.")
