#!/usr/bin/env python3
"""Cluster remarked antibody structures by target-aligned heavy-chain RMSD.

Given a directory that contains remarked H/L/T antibody structures (in either
PDB or mmCIF format), this utility performs the following steps:

1.  Parse the structures and extract Cα coordinates for the target (``T``) and
    heavy (``H``) chains.
2.  For every ordered pair of structures, align the moving target chain onto
    the reference target chain with a closed-form Kabsch superposition and then
    compute the RMSD between the heavy chains under that alignment.
3.  Symmetrise the directional RMSDs to obtain a distance matrix and cluster
    the structures with complete-linkage agglomerative clustering, merging
    clusters whose maximal intra-cluster RMSD does not exceed the requested
    threshold (2 Å by default).
4.  For each cluster, emit a PyMOL script (``.pml``) that loads every member,
    aligns the target chains using the exact residue correspondences employed
    during the RMSD calculation, and orients the view.  If a PyMOL Python
    module (``pymol2``) is available, a ready-to-open ``.pse`` session is also
    produced automatically.

All helper functions live in this file so that the script remains fully
stand-alone and can be copied into other environments without the rest of the
``rfantibody`` package.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


ResidueKey = Tuple[str, int, str]


@dataclass
class ChainCoordinates:
    """Container for a single chain's residue identifiers and coordinates."""

    residue_keys: List[ResidueKey]
    coords: np.ndarray


Structure = Dict[str, ChainCoordinates]


@dataclass
class ParsedStructure:
    """A structure paired with its on-disk location."""

    path: Path
    chains: Structure


def parse_structure(path: Path) -> Structure:
    """Dispatch to the appropriate parser based on the file suffix."""

    suffix = path.suffix.lower()
    if suffix in {".pdb", ".ent", ".brk"}:
        return parse_pdb_structure(path)
    if suffix in {".cif", ".mmcif"}:
        return parse_mmcif_structure(path)
    raise ValueError(f"Unsupported structure format for '{path}'.")


def parse_pdb_structure(pdb_path: Path) -> Structure:
    """Parse a remarked H/L/T PDB file and extract Cα coordinates by chain."""

    per_chain: Dict[str, Dict[ResidueKey, np.ndarray]] = {}
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            altloc = line[16:17]
            if altloc not in (" ", "A", "1"):
                continue
            chain_id = line[21:22].strip() or " "
            seq_field = line[22:26].strip()
            if not seq_field:
                raise ValueError(f"Missing residue number in line: {line.rstrip()}")
            residue_number = int(seq_field)
            insertion_code = line[26:27].strip()
            key: ResidueKey = (chain_id, residue_number, insertion_code)
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid coordinate in line: {line.rstrip()}") from exc
            chain_data = per_chain.setdefault(chain_id, {})
            chain_data.setdefault(key, np.array([x, y, z], dtype=np.float64))

    return finalize_structure(per_chain)


def parse_mmcif_structure(cif_path: Path) -> Structure:
    """Parse a remarked H/L/T mmCIF file and extract Cα coordinates by chain."""

    lines = cif_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    field_names: List[str] = []
    data_start = None
    for idx, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower == "loop_":
            field_names = []
            data_start = None
            continue
        if stripped.startswith("_atom_site."):
            field_names.append(stripped.split(".", 1)[1].strip())
            continue
        if field_names and not stripped.startswith("_atom_site."):
            data_start = idx
            break

    if not field_names or data_start is None:
        raise ValueError(f"No _atom_site loop found in '{cif_path}'.")

    records: List[List[str]] = []
    i = data_start
    num_fields = len(field_names)
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("#"):
            break
        if stripped.startswith("loop_") or stripped.startswith("_"):
            break
        tokens = shlex.split(stripped, comments=False, posix=True)
        while len(tokens) < num_fields:
            i += 1
            if i >= len(lines):
                break
            tokens.extend(shlex.split(lines[i].strip(), comments=False, posix=True))
        if len(tokens) < num_fields:
            break
        records.append(tokens[:num_fields])
        i += 1

    name_to_index = {name: idx for idx, name in enumerate(field_names)}

    def get_value(row: List[str], *candidates: str) -> str | None:
        for candidate in candidates:
            idx = name_to_index.get(candidate)
            if idx is None:
                continue
            value = row[idx]
            if value not in {"?", "."}:
                return value
        return None

    per_chain: Dict[str, Dict[ResidueKey, np.ndarray]] = {}
    for row in records:
        group_pdb = get_value(row, "group_PDB")
        if group_pdb != "ATOM":
            continue

        atom_name = (get_value(row, "auth_atom_id") or get_value(row, "label_atom_id") or "").strip()
        if atom_name.upper() != "CA":
            continue

        alt_id = (get_value(row, "label_alt_id") or " ").strip()
        if alt_id not in {"", " ", "A", "1"}:
            continue

        chain_id = (
            (get_value(row, "auth_asym_id") or get_value(row, "label_asym_id") or "").strip()
            or " "
        )

        seq_id = get_value(row, "auth_seq_id", "label_seq_id")
        if seq_id is None:
            raise ValueError(
                "Missing residue number in mmCIF row for chain "
                f"'{chain_id}' of '{cif_path}'."
            )
        residue_number = int(float(seq_id))
        insertion_code = (get_value(row, "pdbx_PDB_ins_code") or "").strip()

        def parse_float(value: str | None) -> float:
            if value is None:
                raise ValueError("Missing coordinate value in mmCIF row.")
            clean = value.split("(", 1)[0]
            return float(clean)

        x = parse_float(get_value(row, "Cartn_x"))
        y = parse_float(get_value(row, "Cartn_y"))
        z = parse_float(get_value(row, "Cartn_z"))

        key: ResidueKey = (chain_id, residue_number, insertion_code)
        chain_data = per_chain.setdefault(chain_id, {})
        chain_data.setdefault(key, np.array([x, y, z], dtype=np.float64))

    return finalize_structure(per_chain)


def finalize_structure(per_chain: Dict[str, Dict[ResidueKey, np.ndarray]]) -> Structure:
    """Convert a mapping of residue keys to an ordered ``Structure``."""

    structure: Structure = {}
    for chain_id, residues in per_chain.items():
        if not residues:
            continue
        sorted_items = sorted(residues.items(), key=lambda item: (item[0][1], item[0][2]))
        residue_keys = [item[0] for item in sorted_items]
        coords = np.vstack([item[1] for item in sorted_items]).astype(np.float64)
        structure[chain_id] = ChainCoordinates(residue_keys, coords)
    return structure


def combine_chains(structure: Structure, chains: Sequence[str]) -> ChainCoordinates:
    """Concatenate coordinates from multiple chains in the provided order."""

    residue_keys: List[ResidueKey] = []
    coords_list: List[np.ndarray] = []
    for chain_id in chains:
        chain = structure.get(chain_id)
        if chain is None or not chain.coords.size:
            continue
        residue_keys.extend(chain.residue_keys)
        coords_list.append(chain.coords)
    if not residue_keys:
        joined = ", ".join(chains)
        raise ValueError(f"No residues found for chain(s): {joined}")
    coords = np.vstack(coords_list) if coords_list else np.zeros((0, 3), dtype=np.float64)
    return ChainCoordinates(residue_keys, coords)


def get_chain(structure: Structure, chain_id: str) -> ChainCoordinates:
    """Return coordinates for a required chain."""

    chain = structure.get(chain_id)
    if chain is None or not chain.coords.size:
        raise ValueError(f"Chain '{chain_id}' not found or empty in structure.")
    return chain


def match_common_residues(
    reference: ChainCoordinates,
    moving: ChainCoordinates,
    *,
    description: str,
    stream=sys.stderr,
) -> Tuple[List[ResidueKey], np.ndarray, np.ndarray]:
    """Return coordinates for residues that exist in both structures."""

    ref_map = {key: reference.coords[idx] for idx, key in enumerate(reference.residue_keys)}
    mov_map = {key: moving.coords[idx] for idx, key in enumerate(moving.residue_keys)}

    common_keys = [key for key in reference.residue_keys if key in mov_map]
    if not common_keys:
        raise ValueError(f"No overlapping residues found for {description}.")

    missing_in_moving = [key for key in reference.residue_keys if key not in mov_map]
    missing_in_reference = [key for key in moving.residue_keys if key not in ref_map]
    if missing_in_moving:
        formatted = ", ".join(format_residue_key(k) for k in missing_in_moving)
        print(
            f"Warning: ignoring {len(missing_in_moving)} {description} residues absent from moving structure: {formatted}",
            file=stream,
        )
    if missing_in_reference:
        formatted = ", ".join(format_residue_key(k) for k in missing_in_reference)
        print(
            f"Warning: ignoring {len(missing_in_reference)} {description} residues absent from reference structure: {formatted}",
            file=stream,
        )

    ref_coords = np.vstack([ref_map[key] for key in common_keys])
    mov_coords = np.vstack([mov_map[key] for key in common_keys])
    return common_keys, mov_coords, ref_coords


def format_residue_key(key: ResidueKey) -> str:
    chain, number, insertion = key
    if insertion:
        return f"{chain}{number}{insertion}"
    return f"{chain}{number}"


def compute_superposition(moving: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rotation and translation aligning ``moving`` onto ``reference``."""

    if moving.shape != reference.shape or moving.ndim != 2 or moving.shape[1] != 3:
        raise ValueError("Coordinate arrays must share shape (N, 3) for alignment.")
    centroid_moving = moving.mean(axis=0)
    centroid_reference = reference.mean(axis=0)
    moving_centered = moving - centroid_moving
    reference_centered = reference - centroid_reference
    covariance = moving_centered.T @ reference_centered
    U, _, Vt = np.linalg.svd(covariance)
    diag = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        diag[-1] = -1.0
    rotation = U @ np.diag(diag) @ Vt
    translation = centroid_reference - centroid_moving @ rotation
    return rotation, translation


def apply_transform(coords: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply an affine transform defined by ``rotation`` and ``translation``."""

    return coords @ rotation + translation


def calc_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD between two aligned coordinate sets."""

    if coords1.shape != coords2.shape or coords1.ndim != 2:
        raise ValueError("Coordinate arrays must share shape (N, 3) for RMSD calculation.")
    if coords1.shape[0] == 0:
        raise ValueError("No residues available for RMSD calculation.")
    diff = coords1 - coords2
    return float(np.sqrt((diff * diff).sum() / coords1.shape[0]))


def compute_target_aligned_chain_rmsd(moving: Structure, reference: Structure) -> float:
    """Align the moving complex on chain T and compute heavy-chain RMSD."""

    target_moving = get_chain(moving, "T")
    target_reference = get_chain(reference, "T")
    _, moving_target, reference_target = match_common_residues(
        target_reference, target_moving, description="target alignment"
    )

    rotation, translation = compute_superposition(moving_target, reference_target)

    heavy_moving = get_chain(moving, "H")
    heavy_reference = get_chain(reference, "H")
    _, moving_heavy, reference_heavy = match_common_residues(
        heavy_reference, heavy_moving, description="heavy-chain RMSD"
    )

    aligned_heavy = apply_transform(moving_heavy, rotation, translation)
    return calc_rmsd(aligned_heavy, reference_heavy)


def compute_symmetric_rmsd(struct_a: Structure, struct_b: Structure) -> float:
    """Average the directional RMSDs to obtain a symmetric distance."""

    forward = compute_target_aligned_chain_rmsd(struct_a, struct_b)
    backward = compute_target_aligned_chain_rmsd(struct_b, struct_a)
    return 0.5 * (forward + backward)


def build_distance_matrix(structures: Sequence[ParsedStructure]) -> np.ndarray:
    """Compute a symmetric RMSD matrix for all structures."""

    count = len(structures)
    matrix = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            rmsd_value = compute_symmetric_rmsd(structures[i].chains, structures[j].chains)
            matrix[i, j] = matrix[j, i] = rmsd_value
    return matrix


def complete_linkage_clustering(distance_matrix: np.ndarray, threshold: float) -> List[List[int]]:
    """Cluster indices with complete linkage under a distance threshold."""

    clusters: List[List[int]] = [[idx] for idx in range(distance_matrix.shape[0])]
    if distance_matrix.shape[0] <= 1:
        return clusters

    while True:
        best_pair: Tuple[int, int] | None = None
        best_distance = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                max_distance = max(
                    distance_matrix[a, b] for a in clusters[i] for b in clusters[j]
                )
                if max_distance <= threshold and max_distance < best_distance:
                    best_pair = (i, j)
                    best_distance = max_distance
        if best_pair is None:
            break
        i, j = best_pair
        clusters[i].extend(clusters[j])
        clusters[i].sort()
        del clusters[j]
    return clusters


def save_distance_matrix_csv(
    matrix: np.ndarray, labels: Sequence[str], output_path: Path
) -> None:
    """Persist the distance matrix with a header row."""

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file"] + list(labels))
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f"{value:.4f}" for value in row])


def write_cluster_summary(
    clusters: Sequence[Sequence[int]],
    structures: Sequence[ParsedStructure],
    output_path: Path,
) -> None:
    """Write a JSON summary of clusters and their member files."""

    cluster_data = []
    for cluster_id, indices in enumerate(clusters, start=1):
        cluster_data.append(
            {
                "cluster_id": cluster_id,
                "size": len(indices),
                "members": [str(structures[idx].path) for idx in indices],
            }
        )
    output_path.write_text(json.dumps(cluster_data, indent=2), encoding="utf-8")


def sanitize_object_name(cluster_id: int, path: Path, existing: Iterable[str]) -> str:
    """Create a PyMOL-friendly object name derived from the file name."""

    base = path.stem or f"structure_{cluster_id}"
    sanitized = "".join(char if char.isalnum() else "_" for char in base)
    if not sanitized:
        sanitized = f"structure_{cluster_id}"
    candidate = f"cluster{cluster_id}_{sanitized}"
    suffix = 1
    existing_set = set(existing)
    while candidate in existing_set:
        candidate = f"cluster{cluster_id}_{sanitized}_{suffix}"
        suffix += 1
    return candidate


def format_residue_selection(object_name: str, residue_keys: Sequence[ResidueKey]) -> str:
    """Build a PyMOL selection covering the supplied residues' Cα atoms."""

    if not residue_keys:
        raise ValueError("Residue list for PyMOL selection is empty.")
    chain_id = residue_keys[0][0]
    if any(key[0] != chain_id for key in residue_keys):  # pragma: no cover - defensive
        raise ValueError("Residue keys span multiple chains; cannot build selection.")
    resi_parts = [
        f"{number}{insertion}" if insertion else f"{number}" for _, number, insertion in residue_keys
    ]
    resi_expr = "+".join(resi_parts)
    chain_expr = chain_id if chain_id.strip() else '" "'
    selection = f"{object_name} and name CA and resi {resi_expr}"
    if chain_expr:
        selection += f" and chain {chain_expr}"
    return selection


def create_cluster_outputs(
    cluster_id: int,
    indices: Sequence[int],
    structures: Sequence[ParsedStructure],
    output_dir: Path,
) -> None:
    """Generate PyMOL artifacts for a cluster."""

    if not indices:
        return

    object_names: Dict[int, str] = {}
    for idx in indices:
        object_names[idx] = sanitize_object_name(cluster_id, structures[idx].path, object_names.values())

    reference_index = indices[0]
    reference_structure = structures[reference_index].chains
    reference_name = object_names[reference_index]

    alignments: Dict[int, List[ResidueKey]] = {}
    for idx in indices:
        if idx == reference_index:
            continue
        moving_structure = structures[idx].chains
        common_keys, _, _ = match_common_residues(
            get_chain(reference_structure, "T"),
            get_chain(moving_structure, "T"),
            description=f"PyMOL alignment to cluster {cluster_id}",
        )
        alignments[idx] = common_keys

    pml_lines = [
        "reinitialize",
        "set retain_order, 1",
    ]
    for idx in indices:
        path = structures[idx].path.resolve()
        obj_name = object_names[idx]
        pml_lines.append(f'load "{path}", {obj_name}')

    for idx in indices:
        if idx == reference_index:
            continue
        keys = alignments.get(idx)
        if not keys:
            continue
        moving_sel = format_residue_selection(object_names[idx], keys)
        reference_sel = format_residue_selection(reference_name, keys)
        moving_sel_escaped = moving_sel.replace('"', '\\"')
        reference_sel_escaped = reference_sel.replace('"', '\\"')
        pml_lines.append(
            f'pair_fit "{moving_sel_escaped}", "{reference_sel_escaped}"'
        )

    pml_lines.append("orient")
    pse_name = f"cluster_{cluster_id:02d}.pse"
    pml_lines.append(f'save "{pse_name}"')
    pml_path = output_dir / f"cluster_{cluster_id:02d}.pml"
    pml_path.write_text("\n".join(pml_lines) + "\n", encoding="utf-8")

    try:
        from pymol2 import PyMOL  # type: ignore
    except ImportError:
        print(
            f"PyMOL Python module not available; wrote {pml_path.name} to reproduce alignment manually.",
            file=sys.stderr,
        )
        return

    with PyMOL() as pymol:
        cmd = pymol.cmd
        cmd.reinitialize()
        for idx in indices:
            cmd.load(str(structures[idx].path), object_names[idx])
        for idx in indices:
            if idx == reference_index:
                continue
            keys = alignments.get(idx)
            if not keys:
                continue
            moving_sel = format_residue_selection(object_names[idx], keys)
            reference_sel = format_residue_selection(reference_name, keys)
            cmd.pair_fit(moving_sel, reference_sel)
        cmd.orient(" or ".join(object_names[i] for i in indices))
        cmd.save(str(output_dir / pse_name))


def gather_structure_paths(directory: Path, recursive: bool) -> List[Path]:
    """Collect structure files from ``directory``."""

    suffixes = {".pdb", ".ent", ".brk", ".cif", ".mmcif"}
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    paths = [path for path in iterator if path.is_file() and path.suffix.lower() in suffixes]
    paths.sort()
    return paths


def relative_label(base_dir: Path, path: Path) -> str:
    """Return a display label relative to the base directory when possible."""

    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute target-aligned heavy-chain RMSDs for remarked antibody structures, "
            "cluster them with complete linkage, and emit PyMOL visualisation assets."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing remarked PDB/mmCIF files with H/L/T chains.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Complete-linkage RMSD threshold in Å (default: 2.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cluster_output"),
        help="Directory in which to place the matrix, cluster summary, and PyMOL files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search the input directory for structures.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory '{input_dir}' does not exist or is not a directory.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    structure_paths = gather_structure_paths(input_dir, recursive=args.recursive)
    if not structure_paths:
        raise SystemExit("No .pdb or .cif files found in the specified directory.")

    parsed_structures: List[ParsedStructure] = []
    for path in structure_paths:
        print(f"Parsing {path}...")
        chains = parse_structure(path)
        parsed_structures.append(ParsedStructure(path=path, chains=chains))

    labels = [relative_label(input_dir, path) for path in structure_paths]
    print("Computing pairwise RMSDs...")
    distance_matrix = build_distance_matrix(parsed_structures)

    matrix_path = output_dir / "target_aligned_chainH_rmsd_matrix.csv"
    save_distance_matrix_csv(distance_matrix, labels, matrix_path)
    print(f"Saved RMSD matrix to {matrix_path}")

    print("Performing complete-linkage clustering...")
    clusters = complete_linkage_clustering(distance_matrix, args.threshold)
    summary_path = output_dir / "clusters.json"
    write_cluster_summary(clusters, parsed_structures, summary_path)
    print(f"Cluster assignments written to {summary_path}")

    for cluster_id, indices in enumerate(clusters, start=1):
        if not indices:
            continue
        members = ", ".join(labels[idx] for idx in indices)
        print(f"Cluster {cluster_id} (size {len(indices)}): {members}")
        create_cluster_outputs(cluster_id, indices, parsed_structures, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

