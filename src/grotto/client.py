from functools import wraps
from typing import Callable

import joblib
import pandas as pd
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

from caveclient.frameworkclient import CAVEclientFull


def _get_object_methods(obj, exclude_private=True):
    object_methods = [
        method_name for method_name in dir(obj) if callable(getattr(obj, method_name))
    ]
    if exclude_private:
        object_methods = [
            method_name
            for method_name in object_methods
            if not method_name.startswith("_")
        ]
    return object_methods


def _promote_methods(obj, target):
    for method_name in _get_object_methods(obj, exclude_private=True):
        if method_name in _get_object_methods(target, exclude_private=False):
            continue
        method = getattr(obj, method_name)
        setattr(target, method_name, method)
        if hasattr(method, "__annotations__"):
            setattr(target, method_name, method)


def parametrized(dec):
    """This decorator allows you to easily create decorators that take arguments"""
    # REF: https://stackoverflow.com/questions/5929107/decorators-with-parameters

    @wraps(dec)
    def layer(*args, **kwargs):
        @wraps(dec)
        def repl(f):
            return dec(f, *args, **kwargs)

        return repl

    return layer


# decorator for methods of grotto client which dispatches the method to multiple objects
# and chunks the operation into multiple jobs based on a chunk_size parameter
@parametrized
def _dispatch_method(method: Callable, output_format="list"):
    # if self.n_jobs == 1:
    #         tasks = list(zip(*args))
    #         data_by_object = []
    #         for sub_args in tqdm(tasks, desc=method.__name__, disable=self.verbose < 1):
    #             data_by_object.append(method(*sub_args, **kwargs))
    #     else:
    #         tasks = list(
    #             joblib.delayed(method)(*sub_args, **kwargs) for sub_args in zip(*args)
    #         )
    #         n_tasks = len(tasks)
    #         if self.verbose > 0:
    #             with tqdm_joblib(total=n_tasks, desc=method.__name__) as progress_bar:  # noqa: F841
    #                 data_by_object = joblib.Parallel(n_jobs=self.n_jobs)(tasks)
    #         else:
    #             data_by_object = joblib.Parallel(n_jobs=self.n_jobs)(tasks)
    #     return data_by_object

    @wraps(method)
    def wrapper(self, array, **kwargs):
        if self.n_jobs == 1:
            data_by_object = []
            for item in tqdm(array, desc=method.__name__, disable=self.verbose == 0):
                data_by_object.append(method(self, item, **kwargs))
        else:
            tasks = list(joblib.delayed(method)(self, item, **kwargs) for item in array)
            n_tasks = len(tasks)
            with tqdm_joblib(
                total=n_tasks, desc=method.__name__, disable=self.verbose == 0
            ) as progress_bar:  # noqa: F841
                data_by_object = joblib.Parallel(n_jobs=self.n_jobs)(tasks)

        if output_format == "list":
            return data_by_object
        elif output_format == "dict":
            return {item: data for item, data in zip(array, data_by_object)}
        elif output_format == "dataframe":
            new_data = []
            for item, data in zip(array, data_by_object):
                for x in data:
                    new_data.append({"object": item, "data": x})
            df = pd.DataFrame(new_data)
            return df

    return wrapper


class GrottoClient(CAVEclientFull):
    def __init__(self, *args, **kwargs):
        if "n_jobs" in kwargs:
            self.n_jobs = kwargs.pop("n_jobs")
        else:
            self.n_jobs = None

        if "verbose" in kwargs:
            self.verbose = kwargs.pop("verbose")
        else:
            self.verbose = False

        super().__init__(*args, **kwargs)

        # this is _not_ lazy
        super().annotation
        # ignore auth
        super().chunkedgraph
        # super().info
        super().l2cache
        super().materialize
        super().schema
        super().state

        # add all methods from the lazy clients into this class
        _promote_methods(super().annotation, self)
        _promote_methods(super().chunkedgraph, self)
        # _promote_methods(super().info, self)
        _promote_methods(super().l2cache, self)
        _promote_methods(super().materialize, self)
        _promote_methods(super().schema, self)
        _promote_methods(super().state, self)

    # override the lazy properties, make sure they have type hints
    # @property
    # def materialize(self) -> MaterializationClientV3:
    #     return self._materialize

    # @property
    # def chunkedgraph(self) -> ChunkedGraphClientV1:
    #     return self.chunkedgraph

    # @property
    # def l2cache(self) -> L2CacheClientLegacy:
    #     return self.l2cache

    # @property
    # def info(self) -> InfoServiceClientV2:
    #     return self.info

    # def _dispatch_method(self, method, *args, **kwargs) -> list:
    #     if self.n_jobs == 1:
    #         tasks = list(zip(*args))
    #         data_by_object = []
    #         for sub_args in tqdm(tasks, desc=method.__name__, disable=self.verbose < 1):
    #             data_by_object.append(method(*sub_args, **kwargs))
    #     else:
    #         tasks = list(
    #             joblib.delayed(method)(*sub_args, **kwargs) for sub_args in zip(*args)
    #         )
    #         n_tasks = len(tasks)
    #         if self.verbose > 0:
    #             with tqdm_joblib(total=n_tasks, desc=method.__name__) as progress_bar:  # noqa: F841
    #                 data_by_object = joblib.Parallel(n_jobs=self.n_jobs)(tasks)
    #         else:
    #             data_by_object = joblib.Parallel(n_jobs=self.n_jobs)(tasks)
    #     return data_by_object

    def query_table(self, *args, **kwargs):
        kwargs["split_positions"] = True
        kwargs["desired_resolution"] = [1.0, 1.0, 1.0]
        out = self.materialize.query_table(*args, **kwargs)
        col_rename = {}
        for dim in ["x", "y", "z"]:
            if f"pt_position_{dim}" in out.columns:
                col_rename[f"pt_position_{dim}"] = dim
        out.rename(columns=col_rename, inplace=True)
        return out

    def get_l2data(self, *args, **kwargs):
        out = self.l2cache.get_l2data(*args, **kwargs)
        out = pd.DataFrame(out).T
        out["x"] = out["rep_coord_nm"].apply(lambda x: x[0] if x is not None else None)
        out["y"] = out["rep_coord_nm"].apply(lambda x: x[1] if x is not None else None)
        out["z"] = out["rep_coord_nm"].apply(lambda x: x[2] if x is not None else None)
        out.index.name = "level2_id"
        out.index = out.index.astype(int)
        return out

    # @_dispatch_method(output_format="dataframe")
    # def get_leaves_multiple(self, node_id, **kwargs):
    #     return self.get_leaves(node_id, **kwargs)
