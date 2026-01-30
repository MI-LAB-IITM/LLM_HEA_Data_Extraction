import pandas as pd
from modules.formula_parser import parse_formula
from modules.validator import valid_composition

class AlloyCategorizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def categorize(self):
        self.df['Category'] = 'Conventional'

        # Rule 1: Missing values are unconventional
        mask_missing = ((self.df['Alloy'].isna()) | (self.df['Alloy'] == "-"))
        self.df.loc[mask_missing, 'Category'] = 'Unconventional'


        # Rule 2: Elements with numbers are unconventional
        mask_element = ((self.df['Type'] == 'element') & (self.df['Alloy'].str.contains(r'\d', regex=True)))
        self.df.loc[mask_element, 'Category'] = 'Unconventional'

        
        # Rule 3: Parse the remaining Conventional alloys to the validator and decide

        try:
            mask_remainder = (
                (self.df['Category'] == 'Conventional')
                & (~self.df['Alloy'].apply(
                    lambda x: valid_composition(parse_formula(x)[0]) if pd.notna(x) else False
                ))
            )
            self.df.loc[mask_remainder, 'Category'] = 'Unconventional'
        except TypeError as e:
            print()

        return self.df


def categorize_alloy(df: pd.DataFrame) -> pd.DataFrame:
    alloy_categorizer = AlloyCategorizer(df)
    
    return alloy_categorizer.categorize()