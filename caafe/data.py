import pandas as pd
import torch
import numpy as np
import openml
import re
from util.FileHandler import reader_CSV


def get_dataset_classification(dataset, dataset_name, target_attribute, description: str, multiclass=True, shuffled=True):
    dataset = openml.datasets.create_dataset(name=dataset_name,description=description,creator="saeed",
                                             contributor="catdb",collection_date="2024-06-13", language="en",
                                             licence="MIT",data=dataset,default_target_attribute=target_attribute,
                                             ignore_attribute=target_attribute,citation="", attributes='auto')
    print(dataset)
    X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe", target=target_attribute)
    print("==============================")

    # if not multiclass:
    #     X = X[y < 2]
    #     y = y[y < 2]
    #
    # if multiclass and not shuffled:
    #     raise NotImplementedError("This combination of multiclass and shuffling isn't implemented")
    #
    # if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
    #     print("Not a NP Array, skipping")
    #     return None, None, None, None
    #
    # if not shuffled:
    #     sort = np.argsort(y) if y.mean() < 0.5 else np.argsort(-y)
    #     pos = int(y.sum()) if y.mean() < 0.5 else int((1 - y).sum())
    #     X, y = X[sort][-pos * 2:], y[sort][-pos * 2:]
    #     y = torch.tensor(y).reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).float()
    #     X = (
    #         torch.tensor(X)
    #         .reshape(2, -1, X.shape[1])
    #         .transpose(0, 1)
    #         .reshape(-1, X.shape[1])
    #         .flip([0])
    #         .float()
    #     )
    # else:
    #     order = np.arange(y.shape[0])
    #     np.random.seed(13)
    #     np.random.shuffle(order)
    #     X, y = torch.tensor(X[order]), torch.tensor(y[order])
    #
    # return (
    #     X,
    #     y,
    #     list(np.where(categorical_indicator)[0]),
    #     attribute_names + [list(dataset.features.values())[-1].name],
    #     description,
    # )


def load_dataset(
        dataset_name: str = None,
        train_path: str = None,
        test_path: str = None,
        target_attribute: str = None,
        description: str = None,
        multiclass: bool = False,
        shuffled: bool = True):

    description = refactor_openml_description(description)
    train_data = reader_CSV(train_path)
    test_data = reader_CSV(test_path)

    (X_train, y_train, categorical_feats_train, attribute_names_train, description) = get_dataset_classification(
        dataset=train_data,
        description=description,
        target_attribute=target_attribute,
        multiclass=multiclass,
        shuffled=shuffled,
    dataset_name=dataset_name)

    (X_test, y_test, categorical_feats_test, attribute_names_test, description) = get_dataset_classification(
        dataset=test_data,
        description=description,
        target_attribute=target_attribute,
        multiclass=multiclass,
        shuffled=shuffled,
    dataset_name=dataset_name)

    def get_df(X, y, categorical_feats, attribute_names):
        df = pd.DataFrame(data=np.concatenate([X, np.expand_dims(y, -1)], -1), columns=attribute_names)
        cat_features = categorical_feats
        for c in cat_features:
            if len(np.unique(df.iloc[:, c])) > 50:
                cat_features.remove(c)
                continue
            df[df.columns[c]] = df[df.columns[c]].astype("int32")
        return df.infer_objects()

    df_train = get_df(X_train, y_train,categorical_feats_train, attribute_names_train )
    df_test = get_df(X_test, y_test, categorical_feats_test, attribute_names_test)
    df_train.iloc[:, -1] = df_train.iloc[:, -1].astype("category")
    df_test.iloc[:, -1] = df_test.iloc[:, -1].astype("category")

    modifications = {
        "samples_capped": False,
        "classes_capped": False,
        "feats_capped": False,
    }

    ds = [dataset_name, X_train, y_train, categorical_feats_train, attribute_names_train, modifications, description,]
    return ds, df_train, df_test

def refactor_openml_description(description):
    if description is None:
        return ""
    """Refactor the description of an openml dataset to remove the irrelevant parts."""
    splits = re.split("\n", description)
    blacklist = [
        "Please cite",
        "Author",
        "Source",
        "Author:",
        "Source:",
        "Please cite:",
    ]
    sel = ~np.array(
        [
            np.array([blacklist_ in splits[i] for blacklist_ in blacklist]).any()
            for i in range(len(splits))
        ]
    )
    description = str.join("\n", np.array(splits)[sel].tolist())

    splits = re.split("###", description)
    blacklist = ["Relevant Papers"]
    sel = ~np.array(
        [
            np.array([blacklist_ in splits[i] for blacklist_ in blacklist]).any()
            for i in range(len(splits))
        ]
    )
    description = str.join("\n\n", np.array(splits)[sel].tolist())
    return description


def get_X_y(df_train, target_name):
    y = torch.tensor(df_train[target_name].astype(int).to_numpy())
    x = torch.tensor(df_train.drop(target_name, axis=1).to_numpy())

    return x, y

