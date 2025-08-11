import pandas as pd
from sklearn.base import clone
from tsfresh.feature_extraction import extract_features
from tsfresh.transformers import FeatureAugmenter
from tsfresh.utilities.dataframe_functions import roll_time_series


class ForecastRegressor:

    def __init__(self, regressor, tsfresh_transformer: FeatureAugmenter):
        self.regressor = regressor
        self.tsfresh_transformer = tsfresh_transformer

    @property
    def column_id(self) -> str:
        return self.tsfresh_transformer.column_id

    @property
    def column_sort(self) -> str:
        return self.tsfresh_transformer.column_sort

    @property
    def column_value(self) -> str:
        return self.tsfresh_transformer.column_value

    def fit(self, X, y):
        self.tsfresh_transformer.set_timeseries_container(X)
        Xt = pd.DataFrame(index=X[self.column_id].unique())
        Xt = self.tsfresh_transformer.transform(Xt)
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(Xt, y)
        return self

    def predict(self, X: pd.DataFrame, horizon: list) -> pd.DataFrame:

        sequence = X.sort_values(self.column_sort)

        for h in horizon:

            self.tsfresh_transformer.set_timeseries_container(sequence)
            Xt = pd.DataFrame(index=sequence[self.column_id].unique())  # Empty dataframe.
            Xt = self.tsfresh_transformer.transform(Xt)
            output = self.regressor_.predict(Xt)

            # Retrieve last row and update it with output value and current timestamp.
            last = sequence.tail(1).copy()
            last.loc[:, self.column_value] = output
            last.loc[:, self.column_sort] = h

            # Auto-increment index by 1.
            last.index = sequence.index.max() + 1

            # Update sequence.
            sequence = pd.concat([sequence, last]).tail(-1)

        return sequence


class GroupsRegressor:
    def __init__(self, regressor, column_id, column_sort, column_value):
        self.regressor = regressor
        self.column_id = column_id
        self.column_sort = column_sort
        self.column_value = column_value

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        ids_to_fit = list(X.groupby(self.column_id).groups)
        pk = [self.column_id, self.column_sort]
        X = X.sort_values(pk).set_index(pk)
        y = y.sort_values(pk).set_index(pk)
        self.regressors_dict_ = {
            name: clone(self.regressor).fit(X.loc[name], y.loc[name].values.flatten())
            for name in ids_to_fit
        }

        return self

    def get_fitted_regressor(self, id_):
        return self.regressors_dict_.get(id_)

    def predict(self, X: pd.DataFrame):
        ids_to_predict = list(X.groupby(self.column_id).groups)
        pk = [self.column_id, self.column_sort]
        X = X.sort_values(pk).set_index(pk)

        return {
            id_: self.get_fitted_regressor(id_).predict(X.loc[id_])
            for id_ in ids_to_predict
            if id_ in self.regressors_dict_
        }
