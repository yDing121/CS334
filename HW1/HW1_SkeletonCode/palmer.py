"""
Pandas DataFrame Manipulation with Palmer Penguins Dataset
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""
try:
    import numpy as np
    import timeit
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    import pip
    pip.main(['install', 'seaborn', 'matplotlib', 'pandas', 'numpy'])
    import numpy as np
    import timeit
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

def load_csv(inputfile):
    """
    Load the csv as a pandas data frame
    
    Parameters
    ----------
    inputfile : string
        filename of the csv to load

    Returns
    -------
    csvdf : pandas.DataFrame
        return the pandas dataframe with the contents
        from the csv inputfile
    """
    # Assuming that the inputfile actually contains a valid filepath and doesn't require exception handling
    return pd.read_csv(inputfile)


def remove_na(inputdf, colname):
    """
    Remove the rows in the dataframe with NA as values 
    in the column specified.
    
    Parameters
    ----------
    inputdf : pandas.DataFrame
        Input dataframe
    colname : string
        Name of the column to check and remove rows with NA

    Returns
    -------
    outputdf : pandas.DataFrame
        return the pandas dataframe with the modified contents
    """

    return inputdf.dropna(subset=[colname])


def onehot(inputdf, colname):
    """
    Convert the column in the dataframe into a one hot encoding.
    The newly converted columns should be at the end of the data
    frame and you should also drop the original column.
    
    Parameters
    ----------
    inputdf : pandas.DataFrame
        Input dataframe
    colname : string
        Name of the column to one-hot encode

    Returns
    -------
    outputdf : pandas.DataFrame
        return the pandas dataframe with the modified contents
    """
    return pd.get_dummies(inputdf, columns=[colname], dtype=int)


def to_numeric(inputdf):
    """
    Extract all the 
    
    Parameters
    ----------
    inputdf : pandas.DataFrame
        Input dataframe

    Returns
    -------
    outputnp : numpy.ndarray
        return the numeric contents of the input dataframe as a 
        numpy array
    """
    return inputdf.select_dtypes(include=['number']).to_numpy()


def main():
    # Load data
    df = load_csv("data/penguins.csv")

    # # Debug
    # print(df.shape)
    # print(df.isna().sum())

    # Remove NA
    df = remove_na(df, "species")
    # print(df.shape)

    # One hot encoding
    df = onehot(df, "species")
    # print(df.columns)
    # print(df[['species_Chinstrap', 'species_Gentoo', 'species_Adelie']].head(None))
    print(df.head(None))

    # Convert to numeric
    df_np = to_numeric(df)
    print(df_np)


if __name__ == "__main__":
    main()
