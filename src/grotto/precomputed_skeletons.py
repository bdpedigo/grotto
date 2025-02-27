from typing import Optional

import cloudvolume
import numpy as np
import pandas as pd
from caveclient import CAVEclient


def create_skeleton_bucket(
    bucket_path: str, client: CAVEclient, vertex_attributes: list[str]
):
    """
    Generates a bucket with info files for storing precomputed skeletons.

    Parameters
    ----------
    bucket_path :
        The path to the bucket where the skeletons will be stored. Follows the
        cloudvolume conventions, so will likely look like
        "gs://bucket-name/path/to/skeletons".
    client :
        The client to use for getting the base info.
    vertex_attributes :
        The list of attributes to store on the vertices of the skeleton. Radius is
        automatically included. Attributes will be added in the order provided.

    Returns
    -------
    :
        The cloudvolume object for writing skeletons to the bucket.
    :
        The attribute info to be used for each skeleton.
    """
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
    """
    Creates a skeleton object from the provided vertices and edges, and optional
    attributes.

    Parameters
    ----------
    vertices :
        The vertices of the skeleton, provided as an (n,3) array of coordinates.
    edges :
        The edges of the skeleton, provided as an (e,2) array of vertex indices.
    segid :
        The segid to associate with the skeleton.
    vertex_attributes :
        The attributes to store on the vertices of the skeleton. If provided, the
        attribute_info must also be provided.
    attribute_info :
        The information about the attributes to be stored on the vertices.

    Returns
    -------
    :
        The skeleton object.
    """
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
