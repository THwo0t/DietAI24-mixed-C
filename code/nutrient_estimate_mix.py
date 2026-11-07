import argparse
import pandas as pd
import numpy as np
from nutrient_estimate import parse_portion_amount
from rag_portion_size import food_items, food_name

# Paths of the original script
RESULTS_PATH = '../dish_metadata.csv'
NUTRITION_PATH = '../FNDDS/2019-2020 FNDDS At A Glance - FNDDS Nutrient Values.xlsx'
GRAMS = ('g', 'gram', 'grams')

def parse_food_items(food_string):
    """
    Parses a string containing food items and their weights.
    """
    if pd.isna(food_string):  # the grams step did not answer
        return np.nan
    # The user's decision after the third development run: the whole answer, not its first
    # paragraph (split('\n\n')[0]); gpt-4o-2024-08-06 often writes a disclaimer paragraph and
    # the "food: value" lines after a blank line
    if food_string.startswith("I can't help to analyze this"):  # "... this text." or "... this image."
        return np.nan
    lines = food_string.split('\n')

    food_dict = {}
    for line in lines:
        if not line.strip():  # between paragraphs
            continue
        try:
            name, weight = line.rsplit(':', 1)  # R-C8-1: food names keep their colons, as in food_items
            # R-C6-2: grams as one non-negative number ('120.5', '150 g', '1,000'); 'none', a
            # negative number or other text is no weight (r'\d+' read 12.5 as 12, -20 as 20)
            food_dict[name.strip()] = parse_portion_amount(weight, GRAMS)
        except ValueError as e:
            print(f"Error processing line: {line}, {e}")

    return food_dict

def match_food_codes_na(food_string):
    """
    Matches food items to None weights.
    """
    # R-C8-1: the foods of rag_portion_size.food_items, as the grams question listed them
    # (split on '; ' and ': ', names were cut at their own semicolons and foods without the
    # space after the colon were dropped)
    return {name.lower(): np.nan for name, _ in food_items(food_string)}

def match_food_codes(food_string, weight_dict):
    """
    Matches food items to their respective weights
    """
    normalized_weights = {food_name(key).lower(): value for key, value in weight_dict.items()}
    return {name.lower(): (codes, normalized_weights.get(name.lower(), np.nan)) for name, codes in food_items(food_string)}

def calculate_dish_nutrition(food_weights, nutrition_df, col_names_nutrition):
    """
    Calculates the total nutrition for a dish based on food weights and nutrition values.
    """
    nutrition_totals = pd.DataFrame(columns=col_names_nutrition)
    food_code_error = 0
    weight_error = 0
    # R-D7-1: no weight until a food has one; starting from 0 gave 0 g to dishes whose foods all
    # failed, an estimate in D7, which a predicted 0 g still is
    weight_total = np.nan

    for food, (codes, weight) in food_weights.items():
        try:
            codes_float = [float(code) for code in codes]
            food_nutrition = nutrition_df[nutrition_df['Food code'].isin(codes_float)][col_names_nutrition]
            
            if pd.isna(weight):
                weight_error += 1
                print(f"Weight is None for {food}")
                continue
            
            weight_total = weight if pd.isna(weight_total) else weight_total + weight
            if not food_nutrition.empty:
                mean_nutrition = food_nutrition.mean()
                nutrition_for_food = mean_nutrition * (weight / 100)
                nutrition_totals = nutrition_totals.add(pd.DataFrame([nutrition_for_food]), fill_value=0)
            else:  # none of its codes is in FNDDS: counted, the dish is not complete
                print(f"No FNDDS code for {food}: {codes}")
                food_code_error += 1
        except ValueError as e:
            print(f"Error processing food item: {food}, {e}")
            food_code_error += 1
    
    return nutrition_totals, food_code_error, weight_error, weight_total

def process_results(df_results, df_nutrition, col_names_nutrition):
    """
    Processes the results DataFrame to calculate nutrition values for each dish.
    """
    ls_food_code_err = []
    ls_weight_err = []
    ls_ingredient_num = []
    ls_weight_est = []

    for i, row in df_results.iterrows():
        food_string = row['GPTFoodCode']
        weight_dict = row['GPTAmountWeight']
        
        if pd.isna(food_string):  # no food code: nothing to weigh
            ls_food_code_err.append(np.nan)
            ls_weight_err.append(np.nan)
            ls_ingredient_num.append(0)
            ls_weight_est.append(np.nan)
        elif pd.isna(weight_dict):
            food_weights = match_food_codes_na(food_string)
            ls_food_code_err.append(0)
            ls_weight_err.append(len(food_weights))
            ls_ingredient_num.append(len(food_weights))
            ls_weight_est.append(np.nan)
        else:
            food_weights = match_food_codes(food_string, weight_dict)
            nutrition_dish, food_code_error_dish, weight_error_dish, weight_dish = calculate_dish_nutrition(food_weights, df_nutrition, col_names_nutrition)
            ls_food_code_err.append(food_code_error_dish)
            ls_weight_err.append(weight_error_dish)
            ls_ingredient_num.append(len(food_weights))
            ls_weight_est.append(weight_dish)
            
            if not nutrition_dish.empty:
                df_results.loc[i, col_names_nutrition] = nutrition_dish.values[0]

    df_results['FoodCodeError'] = ls_food_code_err
    df_results['WeightError'] = ls_weight_err
    df_results['IngredientNum'] = ls_ingredient_num
    df_results['GPTWeight'] = ls_weight_est

    return df_results

def main(args):
    # Load data
    df_results = pd.read_csv(args.results_path)
    df_nutrition = pd.read_excel(args.nutrition_path, sheet_name='FNDDS Nutrient Values', skiprows=1)
    
    # Parse food items
    df_results['GPTAmountWeight'] = df_results['GPTAmount'].apply(parse_food_items)

    # Initialize columns for nutrition values
    col_names_nutrition = df_nutrition.columns[4:]
    df_results[col_names_nutrition] = pd.DataFrame([[np.nan]*len(col_names_nutrition)], index=df_results.index)

    # Process results
    process_results(df_results, df_nutrition, col_names_nutrition).to_csv(args.output_path or args.results_path, index=False)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Dish nutrients of the Nutrition5k estimates.")
    parser.add_argument("--results_path", default=RESULTS_PATH, help="Read, and written back unless --output_path.")
    parser.add_argument("--output_path")
    parser.add_argument("--nutrition_path", default=NUTRITION_PATH)
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_args())
