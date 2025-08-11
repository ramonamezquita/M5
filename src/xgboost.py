from utils import reduce_mem_usage

import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.feature_extraction import extract_features, MinimalFCParameters
from tsfresh.transformers.feature_augmenter import FeatureAugmenter


def load(filename: str) -> pd.DataFrame:
    return pd.read_csv(f"Dataset/{filename}")


# Load data.
sales_df    = load("sales_train_validation.csv")
prices_df   = load("sell_prices.csv")
calendar_df = load("calendar.csv")


# Add ID column to calendar df.
calendar_df["calendar_id"]  = "d_" + (calendar_df.index + 1).astype(str)


# Keep only two ID_VARS: one for the item and one for the store.
# These two variables contain all the info of the product description.
ID_VARS = ["item_id", "store_id"]
to_drop = ["dept_id", "cat_id", "state_id"]
sales_df.drop(to_drop, axis=1, inplace=True)
melt_df = sales_df.melt(ID_VARS, var_name="d", value_name="target")
melt_df = reduce_mem_usage(melt_df)


# Only forecast top `n` items at the aggregated level.

agg_df = (
    melt_df
    .groupby(["item_id", "d"], as_index=False).sum({"target": "sum"})
    .merge(calendar_df, how="inner", left_on="d", right_on="calendar_id").drop("calendar_id", axis=1)
    .sort_values(["item_id", "date"])
    .reset_index(drop=True)[["item_id", "date", "target"]]
)

n         = 100
top_items = agg_df.groupby("item_id").agg({"target": "sum"})["target"].nlargest(n).index.tolist()


# `input_timeseries_container` contains the timeseries used to train.
input_timeseries_container = agg_df.loc[agg_df["item_id"].isin(top_items)].reset_index(drop=True)


# # tsfresh

MAX_SEQUENCE_LENGTH = 30
MIN_SEQUENCE_LENGTH = 10

# Rolling window.
rolled_df = roll_time_series(
    input_timeseries_container, 
    column_id="item_id", 
    column_sort="date", 
    column_kind="target", 
    max_timeshift=MAX_SEQUENCE_LENGTH, 
    min_timeshift=MIN_SEQUENCE_LENGTH, 
    n_jobs=4
)



# Timeseries features extration.
pk = ["item_id", "date"]

transformer = FeatureAugmenter(
    column_id="id", 
    column_sort="date", 
    column_value="target",
    default_fc_parameters=MinimalFCParameters(),
    timeseries_container=rolled_df.drop("item_id", axis=1)
)

transformed_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(rolled_df.id.unique()))
transformed_df = transformer.transform(transformed_df)
transformed_df.index.set_names(pk, inplace=True)



# Add shifted target values (y) to `transformed_df`.
input_timeseries_container["target__shifted"] = input_timeseries_container.groupby("item_id")["target"].shift(-1)
right = input_timeseries_container.set_index(pk)["target__shifted"].dropna()
transformed_df = transformed_df.merge(right, left_index=True, right_index=True)


# # Fit

from sklearn.ensemble import HistGradientBoostingRegressor

def fit_many(base_regressor, X, y, column_id, column_sort, names = None):
    if names is None:
        names = list(X.groupby(column_id).groups)

    pk = [column_id, column_sort]
    X = X.sort_values(pk).set_index(pk)
    y = y.sort_values(pk).set_index(pk)
    return {name: base_regressor.fit(X.loc[name], y.loc[name].values.flatten()) for name in names}


y = transformed_df.reset_index()[["item_id", "date", "target__shifted"]]
X = transformed_df.reset_index().drop("target__shifted", axis=1)
regressors_dict = fit_many(HistGradientBoostingRegressor(), X, y, column_id="item_id", column_sort="date")


# Predict

def autoregressive_predict(
    predictor, 
    initial_sequence, 
    extractor, 
    column_id,
    column_output, 
    column_sort, 
    horizon
):

    sequence = initial_sequence.sort_values(column_sort)

    for h in horizon:

        extractor.set_timeseries_container(sequence)
        X = pd.DataFrame(index=sequence[column_id].unique())  # Empty dataframe.
        X = extractor.transform(X)
        output = predictor.predict(X)

        # Retrieve last row and update it with output value and current timestamp.
        last = sequence.tail(1).copy()
        last.loc[:, column_output] = output
        last.loc[:, column_sort] = h

        # Auto-increment index by 1.
        last.index += 1

        # Update sequence.
        sequence = pd.concat([sequence, last]).tail(-1)

    return sequence


# In[16]:


# Predict for multiple `item_id`.

import random

all_predictions = []

extractor = FeatureAugmenter(
    column_id="item_id", 
    column_sort="date", 
    column_value="target",
    default_fc_parameters=MinimalFCParameters(),
    disable_progressbar=True
)

ids_to_predict = random.sample(top_items, 10)

for item_id in ids_to_predict:

    if item_id not in regressors_dict:
        continue

    initial_sequence = (
        input_timeseries_container
            .set_index("item_id")
            .loc[item_id]
            .tail(MAX_SEQUENCE_LENGTH)
            .reset_index()
            .drop("target__shifted", axis=1)
    )
    
    initial_sequence["target"] = initial_sequence["target"].astype(float)
    predictor = regressors_dict[item_id]
    train_end = initial_sequence["date"].max()
    forecast_horizon = pd.date_range(train_end, periods=15, inclusive="right", freq="D").astype(str)

    item_id_prediction = autoregressive_predict(
        predictor, initial_sequence, extractor, column_id="item_id", 
        column_output="target", column_sort="date", horizon=forecast_horizon
    )

    all_predictions.append(item_id_prediction)

all_predictions_container = pd.concat(all_predictions)


# In[17]:


input_and_output_container = pd.concat(
    [
        all_predictions_container, 
        input_timeseries_container.loc[input_timeseries_container["item_id"].isin(ids_to_predict)].drop("target__shifted", axis=1)
    ]
).drop_duplicates(["item_id", "date"]).sort_values(["item_id", "date"]).reset_index(drop=True)


# # Plot

# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns

# Ensure date is datetime
input_and_output_container["date"] = pd.to_datetime(input_and_output_container["date"])
forecast_start = forecast_horizon[0]

# Get items to visualize
items_to_plot = ids_to_predict
n_plot = len(items_to_plot )

# Plot
plt.figure(figsize=(16, 5 * n_plot))

for i, item in enumerate(items_to_plot, 1):
    df_item = input_and_output_container[input_and_output_container["item_id"] == item].iloc[-100:].copy()
    df_item["is_forecast"] = df_item["date"] > forecast_start

    plt.subplot(n_plot, 1, i)
    sns.lineplot(data=df_item, x="date", y="target", hue="is_forecast", palette={False: "blue", True: "orange"})
    plt.title(f"Item ID: {item} — Actual vs Forecast")
    plt.xlabel("Date")
    plt.ylabel("Target Value")
    plt.legend(title="Forecast")
    plt.grid(True)

plt.tight_layout()
plt.show()

