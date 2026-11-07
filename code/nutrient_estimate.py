import argparse
import re
import inflect
import pandas as pd
import numpy as np
from asa24_proc import GRAM_PORTION_CODE
from rag_portion_size import extract_amount, extract_unit, normalize_portion, parse_amount, singularize

# File paths
path_portion_code = '../FoodWeights.csv'
path_results = '../ASA24_GPTFoodCodes_portion.csv'
path_nutrition = '../FNDDS/2019-2020 FNDDS At A Glance - FNDDS Nutrient Values.xlsx'
path_image_link = '../df_image_link.csv'
output_path = '../ASA24_GPTFoodCodes_nutrition.csv'

# Load data
def load_nutrition(path):
    df_nutrition = pd.read_excel(path, sheet_name='FNDDS Nutrient Values', skiprows=1)
    col_names_nutrition = df_nutrition.columns[4:]
    return df_nutrition, col_names_nutrition

# Helper function to get portion weight
def get_portion_weight(df_portion_code, food_code, portion_code, portion_subcode):
    if portion_code == GRAM_PORTION_CODE:  # ASA24's 1 g portion, not in FNDDS (B4, R-C6-1)
        return 1.0
    df_filtered = df_portion_code[(df_portion_code['Food Code'] == food_code) & (df_portion_code['Portion Code'] == portion_code)]
    if len(df_filtered) > 1:
        df_filtered = df_filtered[df_filtered['Subcode'] == portion_subcode]
    if len(df_filtered) == 1:
        return df_filtered['Portion Weight'].values[0]
    return np.nan

# Helper function to get nutrition values
def get_nutrition_values(df_nutrition, col_names_nutrition, food_code, weight):
    nutrition_values = {}
    food_code_nutrition = df_nutrition[df_nutrition['Food code'] == food_code]
    if len(food_code_nutrition) == 1 and weight >= 0:
        for nutrition_name in col_names_nutrition:
            nutrition_values[nutrition_name] = food_code_nutrition[nutrition_name].values[0] * weight * 0.01
    else:
        print(f"Food code {food_code} not found in nutrition data")
    return nutrition_values

def map_portion(df_image_link, food_code, description):
    """C6: the (PortionCode, PortionSubCode, Multiplier) of the option the model chose, looked up
    by its text in the portion series of the same code; NaN when the answer is not an option."""
    if pd.isna(food_code) or pd.isna(description):
        return np.nan, np.nan, np.nan
    series = df_image_link[df_image_link['FoodCode'] == food_code]
    rows = series[series['Portion'].map(normalize_portion) == normalize_portion(description)]
    if rows.empty:
        return np.nan, np.nan, np.nan
    return rows['PortionCode'].iloc[0], rows['PortionSubCode'].iloc[0], rows['Multiplier'].iloc[0]

# R-C6-2: an answer is one non-negative number, optionally followed by its unit: '2', '1.5 cups',
# '1/2', '1-1/2', '1 1/2', '1,000'. Text before the number, a second number, a negative number
# or a refusal is no answer.
NUMBER = r'\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.\d+|\d+[- ]\d+/\d+|\d+/\d+|\d+'
NUMBER_ANSWER = re.compile(rf'({NUMBER})([^\d\n]*)')

def parse_portion_amount(answer, units=None):
    """The number of an amount answer; NaN when the answer is not a number as above, or when
    `units` is given and the answer names another unit."""
    if isinstance(answer, (int, float)):
        return float(answer) if np.isfinite(answer) and answer >= 0 else np.nan
    match = NUMBER_ANSWER.fullmatch(str(answer).strip())
    if not match:
        return np.nan
    unit = re.match(r'\W*([A-Za-z]*)', match.group(2)).group(1).lower()
    if units and unit and unit not in units:
        return np.nan
    return parse_amount(match.group(1).replace(',', '').replace(' ', '-'))

def unit_weight(df_image_link, food_code, unit, p):
    """Grams of one unit (e.g. 'cup') of the food, from the options of its series in that unit."""
    series = df_image_link[df_image_link['FoodCode'] == food_code]
    grams = []
    for portion, weight in zip(series['Portion'], series['Weight']):
        amount = parse_amount(extract_amount(portion))
        if singularize(extract_unit(portion), p) == unit and amount > 0:
            grams.append(weight / amount)
    return float(np.median(grams)) if grams else np.nan

def estimate(df_results, df_portion_code, df_image_link, df_nutrition, col_names_nutrition):
    """TotalWeight and nutrients from the chosen option (C6, C7); AmountWeight, the weight read
    from the amount answer, for the second reading of D2."""
    p = inflect.engine()
    # Initialize columns in df_results
    df_results['TotalWeight'] = np.nan
    for nutrient_name in col_names_nutrition:
        df_results[nutrient_name] = np.nan
    for column in ('GPTPortionCode', 'GPTPortionSubCode', 'GPTMultiplier', 'AmountWeight'):
        df_results[column] = np.nan

    # Process each row in df_results
    for i in range(len(df_results)):
        food_code = df_results.loc[i, 'FoodCodeCommon']
        if pd.isna(food_code):
            continue
        # The predicted portion; PortionCode and PortionSubCode hold the label (B4)
        portion_code, portion_subcode, multiplier = map_portion(df_image_link, food_code,
                                                                df_results.loc[i, 'GPTPortionDescription'])
        df_results.loc[i, ['GPTPortionCode', 'GPTPortionSubCode', 'GPTMultiplier']] = [portion_code, portion_subcode, multiplier]

        # The image shows Multiplier x the portion code's weight (ASA24 metadata); the original
        # looked up one unit of the portion code only
        weight = multiplier * get_portion_weight(df_portion_code, food_code, portion_code, portion_subcode)
        df_results.loc[i, 'TotalWeight'] = weight

        nutrition_values = get_nutrition_values(df_nutrition, col_names_nutrition, food_code, weight)
        for nutrient_name, value in nutrition_values.items():
            df_results.loc[i, nutrient_name] = value

        amount = parse_portion_amount(df_results.loc[i, 'GPTPortionAmount'])
        df_results.loc[i, 'AmountWeight'] = amount * unit_weight(df_image_link, food_code, df_results.loc[i, 'LabelUnit'], p)
    return df_results

def end_to_end_weight(df_results, df_portion_code):
    """D3: the amount answer x the FNDDS weight of the chosen portion of GPTFoodCodeE2E."""
    weights = []
    for code, description, amount in zip(df_results['GPTFoodCodeE2E'], df_results['GPTPortionDescriptionE2E'],
                                         df_results['GPTPortionAmountE2E']):
        rows = df_portion_code[(df_portion_code['Food Code'] == code)
                               & (df_portion_code['Portion Description'].map(normalize_portion) == normalize_portion(description))]
        weights.append(parse_portion_amount(amount) * rows['Portion Weight'].iloc[0] if len(rows) and not pd.isna(description) else np.nan)
    df_results['WeightE2E'] = weights
    return df_results

def main(args):
    df_portion_code = pd.read_csv(args.portion_code_path)
    df_results = pd.read_csv(args.results_path)
    df_image_link = pd.read_csv(args.image_link_path)
    df_nutrition, col_names_nutrition = load_nutrition(args.nutrition_path)
    df_results = estimate(df_results, df_portion_code, df_image_link, df_nutrition, col_names_nutrition)
    if 'GPTFoodCodeE2E' in df_results.columns:
        df_results = end_to_end_weight(df_results, df_portion_code)
    # Save the updated dataframe
    df_results.to_csv(args.output_path, index=False)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Weights and nutrients of the ASA24 portion estimates.")
    parser.add_argument("--portion_code_path", default=path_portion_code)
    parser.add_argument("--results_path", default=path_results)
    parser.add_argument("--nutrition_path", default=path_nutrition)
    parser.add_argument("--image_link_path", default=path_image_link)
    parser.add_argument("--output_path", default=output_path)
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_args())
