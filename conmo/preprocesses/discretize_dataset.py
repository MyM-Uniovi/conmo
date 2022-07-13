from typing import Iterable, Union

import pandas as pd

from conmo.conf import Index
from conmo.preprocesses.preprocess import ExtendedPreprocess


class DiscretizeDataset(ExtendedPreprocess):

    def __init__(self, to_data: Union[bool, Iterable[str]], to_labels: Union[bool, Iterable[str]], test_set: bool) -> None:
        super().__init__(to_data, to_labels, test_set)

    def transform(self, df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        if self.test_set == True:
            # Select all sequences
            groups = df.groupby(
                level=[Index.FOLD, Index.SET, Index.SEQUENCE])
        else:
            # Select only TRAIN sequences
            groups = df.loc[(slice(None), Index.SET_TRAIN), :].groupby(
                level=[Index.FOLD, Index.SET, Index.SEQUENCE])
        
        # Apply to each sequence
        for column in columns:
            col_d = pd.Series()
            for group in groups:
                max_v = group[1].loc[group[0]:,column].max()
                min_v = group[1].loc[group[0]:,column].min()
                #print(col_name) # Debug purposes
                aux = df.loc[group[0], column].apply(lambda x: str(chr(65+int((x-min_v)/(max_v-min_v)*4))))
                col_d = col_d.append(aux)
            df.loc[:, column] = col_d.to_numpy()
        
        # Combine all the columns into one column
        df = df.apply(lambda x: ''.join(x.astype(str)), axis=1)

        return pd.DataFrame(df, columns=['discretized_data'])
