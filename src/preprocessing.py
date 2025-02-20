import pandas as pd
import numpy as np
import ast
import pandas as pd

class DataPreprocessor:
    @staticmethod
    def read_csv(file_path: str) -> pd.DataFrame:
        """Loads a CSV file and returns a DataFrame."""
        df = pd.read_csv(file_path)
        print(f"Loaded dataset from '{file_path}'.")
        return df

    @staticmethod
    def check_nonnegative_integer(df: pd.DataFrame, col: str) -> float:
        """Checks if all values in the column are non-negative integers."""
        if col not in df.columns:
            print(f"Column '{col}' not found.")
            return 0
        percentage = (df[col].apply(lambda x: isinstance(x, int) and x >= 0).mean()) * 100
        print(f"{100 - percentage:.2f}% of '{col}' are not non-negative integers.")
        return percentage

    @staticmethod
    def check_for_nans(df: pd.DataFrame, col: str) -> float:
        """Checks the percentage of NaN values in the column."""
        if col not in df.columns:
            print(f"Column '{col}' not found.")
            return 0
        percentage = df[col].isna().mean() * 100
        print(f"'{col}' contains {percentage:.2f}% NaNs.")
        return percentage

    @staticmethod
    def check_int_1900_to_2025(df: pd.DataFrame, col: str) -> float:
        """Checks if values in the column are integers between 1900 and 2025."""
        if col not in df.columns:
            print(f"Column '{col}' not found.")
            return 0
        percentage = (df[col].apply(lambda x: isinstance(x, int) and 1900 <= x <= 2025).mean()) * 100
        print(f"{100 - percentage:.2f}% of '{col}' are out of range 1900-2025.")
        return percentage

    @staticmethod
    def validate_columns(df: pd.DataFrame):
        """Validates 'year' and 'n_citation' columns if they exist."""
        rules = {"year": DataPreprocessor.check_int_1900_to_2025, "n_citation": DataPreprocessor.check_nonnegative_integer}
        for col, func in rules.items():
            if col in df.columns:
                func(df, col)
            else:
                print(f"Column '{col}' not found.")

    @staticmethod
    def save_df_to_csv(df: pd.DataFrame, path: str, filename: str):
        """Saves the DataFrame as a CSV file."""
        if not filename.endswith('.csv'):
            raise ValueError("Filename must end with '.csv'.")
        full_path = f"{path}/{filename}"
        df.to_csv(full_path, index=False)
        print(f"Saved as '{full_path}'")

    @staticmethod

    def process_dataframe_references(df: pd.DataFrame) -> pd.DataFrame:
        """Process and filter references in the DataFrame.
        
        Keeps only rows where all references are valid (appear in the 'id' column).
        """
        
        # Convert references from string to list if needed, stripping whitespace first.
        df['references'] = df['references'].apply(
            lambda x: ast.literal_eval(x.strip()) if isinstance(x, str) and x.strip().startswith("[") else x
        )
        
        # Ensure that references is a list; if not, replace with an empty list.
        df['references'] = df['references'].apply(lambda x: x if isinstance(x, list) else [])
        
        # Normalize paper IDs: strip extra whitespace.
        paper_ids = set(x.strip() for x in df['id'].dropna().astype(str))
        
        print("Sample references:", df['references'].head())
        print("Paper IDs sample:", list(paper_ids)[:10])
        
        def filter_existing_references(ref_list):
            """Return the list of references only if all references are present in the dataset.
            
            If the list is empty or if any reference is invalid, return np.nan.
            """
            if not ref_list:
                return np.nan
            # Clean and check each reference.
            cleaned_refs = [str(ref).strip() for ref in ref_list]
            # Check if every reference in the cleaned list is in the set of valid paper_ids.
            if all(ref in paper_ids for ref in cleaned_refs):
                return cleaned_refs
            else:
                return np.nan
        
        # Apply the filtering function.
        df['filtered_references'] = df['references'].apply(filter_existing_references)
        
        # Remove rows where not all references are valid.
        df = df[df['filtered_references'].notna()]
        
        total_papers_with_refs = df['references'].apply(lambda x: len(x) > 0).sum()
        papers_with_all_valid_refs = df['filtered_references'].apply(lambda x: isinstance(x, list) and len(x) > 0).sum()
        
        print(f"Number of papers with all valid references: {papers_with_all_valid_refs} / {total_papers_with_refs}")
        
        return df


    @staticmethod
    def format_abstracts(df: pd.DataFrame) -> pd.DataFrame:
        """Format abstracts by merging title and venue into a single column."""
        
        df['abstract'] = df.apply(
            lambda row: f"Title: {row['title']}\n\nVenue: {row['venue']}\n\nAbstract: {row['abstract']}", axis=1
        )
        
        df.drop(columns=['title', 'venue'], inplace=True)
        
        return df