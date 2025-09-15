#!/usr/bin/env python
"""Align remarked antibody complexes on the target chain and report RMSD.

This stand-alone utility loads two PDB files in remarked H/L/T format, aligns the
moving structure's target chain onto the reference target using a Kabsch
superposition, and then calculates the root-mean-square deviation (RMSD) between
the antibody chains (H and L) after the alignment. Only Cα atoms are used for
both the alignment and the RMSD computation.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ResidueKey = Tuple[str, int, str]


@dataclass
class ChainCoordinates:
    """Container for a single chain's residue identifiers and coordinates."""

    residue_keys: List[ResidueKey]
    coords: np.ndarray


Structure = Dict[str, ChainCoordinates]


def parse_remarked_structure(pdb_path: Path) -> Structure:
    """Parse a remarked H/L/T PDB file and extract Cα coordinates by chain."""

    per_chain: Dict[str, Dict[ResidueKey, np.ndarray]] = {}
    with pdb_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            altloc = line[16:17]
            if altloc not in (" ", "A"):
                continue
            chain_id = line[21:22].strip() or " "
            seq_field = line[22:26].strip()
            if not seq_field:
                raise ValueError(f"Missing residue number in line: {line.rstrip()}" )
            residue_number = int(seq_field)
            insertion_code = line[26:27].strip()
            key: ResidueKey = (chain_id, residue_number, insertion_code)
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            chain_data = per_chain.setdefault(chain_id, {})
            chain_data.setdefault(key, np.array([x, y, z], dtype=np.float64))

    structure: Structure = {}
    for chain_id, residues in per_chain.items():
        if not residues:
            continue
        sorted_items = sorted(residues.items(), key=lambda item: (item[0][1], item[0][2]))
        keys = [item[0] for item in sorted_items]
        coords = np.vstack([item[1] for item in sorted_items]).astype(np.float64)
        structure[chain_id] = ChainCoordinates(keys, coords)
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
    stream = sys.stderr,
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


def compute_target_aligned_antibody_rmsd(
    moving_structure: Path, reference_structure: Path
) -> float:
    """Align the moving complex on the target and compute antibody RMSD."""

    moving = parse_remarked_structure(moving_structure)
    reference = parse_remarked_structure(reference_structure)

    target_moving = get_chain(moving, "T")
    target_reference = get_chain(reference, "T")
    _, moving_target, reference_target = match_common_residues(
        target_reference, target_moving, description="target alignment"
    )

    rotation, translation = compute_superposition(moving_target, reference_target)

    antibody_moving = combine_chains(moving, ("H", "L"))
    antibody_reference = combine_chains(reference, ("H", "L"))
    _, moving_antibody, reference_antibody = match_common_residues(
        antibody_reference, antibody_moving, description="antibody RMSD"
    )

    aligned_antibody = apply_transform(moving_antibody, rotation, translation)
    return calc_rmsd(aligned_antibody, reference_antibody)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align two remarked PDB structures on the target chain and compute "
            "the antibody RMSD."
        )
    )
    parser.add_argument(
        "moving",
        type=Path,
        help=(
            "PDB file to align onto the reference. Must contain remarked H/L/T "
            "chains."
        ),
    )
    parser.add_argument(
        "reference",
        type=Path,
        help=(
            "Reference remarked PDB file that provides the target coordinates "
            "for alignment."
        ),
    )
    return parser.parse_args(args)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rmsd_value = compute_target_aligned_antibody_rmsd(args.moving, args.reference)
    print(f"target_aligned_antibody_rmsd: {rmsd_value:.3f} Å")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
