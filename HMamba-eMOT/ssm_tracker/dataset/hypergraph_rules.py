"""Rule-based incidence matrices for frame-level multi-relation hypergraphs."""

import numpy as np


def _components_from_similarity(features, threshold, topk):
    n = len(features)
    groups = []
    if n < 2:
        return groups
    features = features.astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    similarity = features @ features.T
    work = similarity.copy()
    np.fill_diagonal(work, -np.inf)
    k = min(int(topk), n - 1)
    nearest = np.argpartition(work, -k, axis=1)[:, -k:]
    nearest_mask = np.zeros((n, n), dtype=bool)
    nearest_mask[np.arange(n)[:, None], nearest] = True
    adjacency = (similarity >= threshold) & nearest_mask & nearest_mask.T
    np.fill_diagonal(adjacency, True)

    visited = np.zeros(n, dtype=bool)
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        stack, component = [start], []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in np.flatnonzero(adjacency[node]):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
        if len(component) >= 2:
            groups.append(np.asarray(component, dtype=np.int64))
    return groups


def _groups_to_incidence(groups, num_vertices, max_edges):
    incidence = np.zeros((num_vertices, max_edges), dtype=np.float32)
    for edge_index, group in enumerate(groups[:max_edges]):
        incidence[group, edge_index] = 1.0
    return incidence


def _upper_pairs(adjacency):
    groups = []
    rows, cols = np.nonzero(np.triu(adjacency, k=1))
    for i, j in zip(rows.tolist(), cols.tolist()):
        groups.append(np.asarray([i, j], dtype=np.int64))
    return groups


def _pair_groups_from_similarity(features, threshold, topk):
    """Undirected pairs that pass the threshold and are mutual top-k neighbors."""
    n = len(features)
    if n < 2:
        return []
    features = features.astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    similarity = features @ features.T
    work = similarity.copy()
    np.fill_diagonal(work, -np.inf)
    k = min(int(topk), n - 1)
    nearest = np.argpartition(work, -k, axis=1)[:, -k:]
    nearest_mask = np.zeros((n, n), dtype=bool)
    nearest_mask[np.arange(n)[:, None], nearest] = True
    adjacency = (similarity >= threshold) & nearest_mask & nearest_mask.T
    np.fill_diagonal(adjacency, False)
    return _upper_pairs(adjacency)


def build_rule_incidence(
    centers,
    rgb_features,
    evt_features,
    motions,
    max_vertices,
    spatial_radius=0.10,
    rgb_threshold=0.80,
    evt_threshold=0.80,
    topk=5,
    pairwise=False,
):
    """Build padded spatial/RGB/Event/motion incidence matrices.

    ``pairwise`` stores each accepted relation as a 2-node edge. The default
    keeps hyperedges: a spatial ball per track, and one edge per connected
    component of the appearance graph.
    """
    n = len(centers)

    center_distance = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=2
    )
    if pairwise:
        spatial_adjacency = center_distance <= spatial_radius
        np.fill_diagonal(spatial_adjacency, False)
        spatial_groups = _upper_pairs(spatial_adjacency)
        rgb_groups = _pair_groups_from_similarity(rgb_features, rgb_threshold, topk)
        evt_groups = _pair_groups_from_similarity(evt_features, evt_threshold, topk)
        relation_slots = max(max_vertices * (max_vertices - 1) // 2, 1)
    else:
        spatial_groups = [
            np.flatnonzero(center_distance[i] <= spatial_radius)
            for i in range(n)
        ]
        spatial_groups = [group for group in spatial_groups if len(group) >= 2]
        rgb_groups = _components_from_similarity(rgb_features, rgb_threshold, topk)
        evt_groups = _components_from_similarity(evt_features, evt_threshold, topk)
        relation_slots = max_vertices

    speed = np.linalg.norm(motions[:, :2], axis=1)
    motion_groups = []
    if n >= 2:
        if n >= 3:
            boundaries = np.quantile(speed, [1.0 / 3.0, 2.0 / 3.0])
            labels = np.digitize(speed, boundaries)
            motion_groups = [
                np.flatnonzero(labels == index)
                for index in range(3)
                if np.sum(labels == index) >= 2
            ]
        else:
            motion_groups = [np.arange(n)]

    return {
        "spatial": _groups_to_incidence(spatial_groups, max_vertices, relation_slots),
        "rgb": _groups_to_incidence(rgb_groups, max_vertices, relation_slots),
        "evt": _groups_to_incidence(evt_groups, max_vertices, relation_slots),
        "motion": _groups_to_incidence(motion_groups, max_vertices, 3),
    }


RELATIONS = ("spatial", "rgb", "evt", "motion")


def relations_to_drop(cfgs):
    """Relations removed for a leave-one-out or single-relation ablation."""
    drop = list(cfgs.get("ablate_drop_relations") or [])
    keep = cfgs.get("ablate_keep_relations")
    if isinstance(drop, str):
        drop = [drop]
    if isinstance(keep, str):
        keep = [keep]
    if keep:
        drop = list(set(drop) | (set(RELATIONS) - set(keep)))
    unknown = [name for name in drop if name not in RELATIONS]
    if unknown:
        raise ValueError(f"unknown hyper relations to drop: {unknown}")
    return drop


def ablate_incidence(incidence, cfgs):
    """Zero whole relation matrices so that relation is absent while training."""
    drop = relations_to_drop(cfgs)
    if not drop:
        return incidence
    missing = [name for name in drop if name not in incidence]
    if missing:
        raise ValueError(f"incidence has no relation {missing}")
    for relation in drop:
        incidence[relation] = np.zeros_like(incidence[relation])
    return incidence
