# -*- coding: utf-8 -*-

def get_std(data, drop_quantile=0.9, for_columns=['re74', 're75']):
    std = data.std()
    for column in for_columns:
        boundary = data[column].quantile(drop_quantile)
        valid = data.loc[data[column] <= boundary, column]
        std[column] = valid.std()
    return std
    
@dataclass
class MyData():
    path: Path
    raw: pd.DataFrame
    scaling: pd.Series

    def __init__(path: Path):
    names = Box(
        x_untreated_full="X0",
        x_treated_full="X1",
        y_untreated_full="Y0",
        y_treated_full="Y1",
        x_untreated_unscaled_full="X0_unscaled"
    )
    result = Box()
    for name, file in names.items():
        file_path = path / (file + ".txt")
        result[name] = pd.read_csv(file_path, sep=" ")
    return result

    @property
    def x_untreated():
        pass

    @property
    def untreated():
        pass
