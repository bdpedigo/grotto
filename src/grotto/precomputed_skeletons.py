from typing import Optional

import cloudvolume
import numpy as np
import pandas as pd
from caveclient import CAVEclient


def create_skeleton_bucket(
    bucket_path: str, client: CAVEclient, vertex_attributes: list[str]
):
    base_cv = client.info.segmentation_cloudvolume()

    info = base_cv.info.copy()

    info["skeletons"] = "skeletons"

    cv = cloudvolume.CloudVolume(
        "precomputed://" + bucket_path,
        info=info,
        compress=False,
    )
    cv.commit_info()

    sk_info = cv.skeleton.meta.default_info()

    attribute_info = [{"id": "radius", "data_type": "float32", "num_components": 1}]
    for attribute in vertex_attributes:
        attribute_info.append(
            {
                "id": attribute,
                "data_type": "float32",
                "num_components": 1,
            }
        )
    sk_info["vertex_attributes"] = attribute_info
    cv.skeleton.meta.info = sk_info
    cv.skeleton.meta.commit_info()
    return cv, attribute_info


def create_skeleton(
    vertices: np.ndarray,
    edges: np.ndarray,
    segid: Optional[int] = None,
    vertex_attributes: Optional[pd.DataFrame] = None,
    attribute_info: dict = None,
):
    skeleton = cloudvolume.Skeleton(
        vertices=vertices.astype(np.float32),
        edges=edges,
        radii=np.ones(len(vertices), dtype=np.float32),
        segid=segid,
        vertex_types=None,
        extra_attributes=attribute_info,
    )
    if vertex_attributes is not None and attribute_info is not None:
        for attribute in attribute_info[1:]:
            skeleton.__setattr__(
                attribute["id"],
                vertex_attributes[attribute["id"]].values.astype(np.float32),
            )
    return skeleton
