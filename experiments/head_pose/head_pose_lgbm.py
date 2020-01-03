import lightgbm as lgbm
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def load_data():
    real_df = pd.read_csv(
        "feat/real.txt",
        delimiter=" ",
        header=None,
        names=["a", "b", "c", "d", "e", "f"],
        index_col=False,
        float_precision="high",
    )

    real_df["target"] = 1

    fake_df = pd.read_csv(
        "feat/fake.txt",
        delimiter=" ",
        header=None,
        names=["a", "b", "c", "d", "e", "f"],
        index_col=False,
        float_precision="high",
    )
    fake_df["target"] = 0

    real_df.head()

    df = pd.concat([real_df, fake_df], ignore_index=True, sort=False)

    del real_df, fake_df

    y = df.target.values
    df = df.drop("target", axis="columns").values
    return df, y


data, target = load_data()

NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)
learning_rate = 0.1
num_leaves = 15
min_data_in_leaf = 2000
feature_fraction = 0.6
num_boost_round = 100
params = {"objective": "binary",
          "boosting_type": "gbdt",
          "learning_rate": learning_rate,
          "num_leaves": num_leaves,
          "max_bin": 256,
          "feature_fraction": feature_fraction,
          "drop_rate": 0.1,
          "is_unbalance": False,
          "max_drop": 50,
          "min_child_samples": 10,
          "min_child_weight": 150,
          "min_split_gain": 0,
          "subsample": 0.9,
          "metric": 'binary',
          "verbose": 5,
          "n_jobs": -1
          }


x_score = []
final_cv_train = np.zeros(len(data))

kf = kfold.split(data, target)


for i, (train_fold, validate) in enumerate(kf):
    X_train, X_validate, label_train, label_validate = \
        data[train_fold, :], data[validate, :], target[train_fold], target[validate]
    dtrain = lgbm.Dataset(X_train, label_train)
    dvalid = lgbm.Dataset(X_validate, label_validate, reference=dtrain)
    bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=dvalid,
                     early_stopping_rounds=100)
    preds = bst.predict(X_validate)
    print(preds.shape)
    print(preds)
    print(f"Fold {i+1} score {accuracy_score(label_validate, preds)}")
