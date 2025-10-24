"""Utility to convert antibody PDB files into the HLT convention.

This variant expects the input PDB to already have each chain numbered
sequentially starting at 1. All configuration, including chain mapping
and CDR definitions, is provided through command line arguments to avoid
interactive prompts.

Example
-------
```
python scripts/util/chothia2HLT.py \
    path/to/input.pdb \
    --heavy A \
    --light B \
    --target C \
    --cdr H:CDR-H1=31-35,40 \
    --cdr H:CDR-H2=50-65 \
    --cdr L:CDR-L1=24-34
```

The command above maps chain ``A`` to ``H``, chain ``B`` to ``L`` and
chain ``C`` to ``T`` while annotating the supplied residue positions as
CDR loops in the output PDB remarks.
"""

import argparse
from typing import Dict, Iterable, List, Tuple

import numpy as np
from biotite.structure import array
from biotite.structure import residue_iter
from biotite.structure.io.pdb import PDBFile

PROTEIN_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the conversion utility."""

    parser = argparse.ArgumentParser(
        description=(
            "Convert a sequentially numbered antibody PDB into HLT format "
            "while annotating user-specified CDR residues."
        )
    )
    parser.add_argument("input_pdb", help="Path to the input PDB file")
    parser.add_argument(
        "--heavy",
        "-H",
        help="Chain identifier in the input file to remap to the H chain",
    )
    parser.add_argument(
        "--light",
        "-L",
        help="Chain identifier in the input file to remap to the L chain",
    )
    parser.add_argument(
        "--target",
        "-T",
        action="append",
        default=[],
        help=(
            "Chain identifier in the input file to remap to the T chain. "
            "Specify multiple times for multiple target chains."
        ),
    )
    parser.add_argument(
        "--cdr",
        action="append",
        default=[],
        metavar="CHAIN:NAME=POSITIONS",
        help=(
            "CDR definition in the form CHAIN:NAME=positions. Positions are "
            "1-indexed and may include comma-separated integers or ranges. "
            "Provide multiple --cdr arguments for additional definitions."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to the output PDB file (defaults to <input>_HLT.pdb)",
    )

    args = parser.parse_args()

    if not args.heavy and not args.light:
        parser.error("At least one of --heavy or --light must be provided.")

    try:
        _validate_cdr_args(args.cdr)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def _validate_cdr_args(cdr_args: Iterable[str]) -> None:
    """Validate that the provided CDR arguments follow the expected format."""

    for item in cdr_args:
        if ":" not in item or "=" not in item:
            raise ValueError(
                "CDR definitions must follow the CHAIN:NAME=POSITIONS format."
            )
        chain_part, remainder = item.split(":", 1)
        name_part, _ = remainder.split("=", 1)
        if not chain_part.strip():
            raise ValueError("CDR definitions require a non-empty chain specifier.")
        if chain_part.strip() not in {"H", "L", "T"}:
            raise ValueError("CDR definitions must target the H, L, or T chain.")
        if not name_part.strip():
            raise ValueError("CDR definitions require a non-empty name.")


def parse_position_list(raw: str) -> List[int]:
    """Parse a comma-separated list of residue positions with optional ranges."""

    positions = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if start <= 0 or end <= 0:
                raise ValueError("Residue numbers must be positive.")
            if end < start:
                raise ValueError("Range end must be >= range start.")
            positions.update(range(start, end + 1))
        else:
            value = int(token)
            if value <= 0:
                raise ValueError("Residue numbers must be positive.")
            positions.add(value)
    if not positions:
        raise ValueError("CDR definitions must specify at least one residue.")
    return sorted(positions)


def parse_cdr_definitions(raw_definitions: Iterable[str]) -> Dict[str, Dict[str, List[int]]]:
    """Parse CDR definitions supplied on the command line."""

    parsed: Dict[str, Dict[str, List[int]]] = {"H": {}, "L": {}, "T": {}}
    for definition in raw_definitions:
        chain_part, remainder = definition.split(":", 1)
        name_part, positions_part = remainder.split("=", 1)
        chain = chain_part.strip()
        name = name_part.strip()
        try:
            positions = parse_position_list(positions_part)
        except ValueError as exc:
            raise ValueError(f"Invalid CDR definition '{definition}': {exc}") from exc
        parsed.setdefault(chain, {})[name] = positions
    return {chain: defs for chain, defs in parsed.items() if defs}


def convert_to_hlt(
    input_pdb: str,
    heavy_chain: str | None,
    light_chain: str | None,
    target_chains: Iterable[str],
    cdr_definitions: Dict[str, Dict[str, List[int]]],
) -> Tuple[np.ndarray, Dict[str, List[int]]]:
    """Convert an input PDB file to HLT format."""

    pdb_file = PDBFile.read(input_pdb)
    structure = pdb_file.get_structure(model=1)

    protein_atom_list = [atom for atom in structure if atom.res_name in PROTEIN_RESIDUES]
    structure = array(protein_atom_list)

    atom_list = []
    cdr_residues: Dict[str, List[int]] = {
        f"{chain}:{cdr_name}": []
        for chain, defs in cdr_definitions.items()
        for cdr_name in defs
    }

    residue_counter = 1

    def process_chain(orig_chain: str | None, new_chain: str) -> None:
        nonlocal residue_counter
        if orig_chain is None:
            return
        chain_mask = structure.chain_id == orig_chain
        if not np.any(chain_mask):
            return
        atoms = structure[chain_mask]
        atoms.chain_id = np.full(len(atoms), new_chain)
        chain_cdrs = cdr_definitions.get(new_chain, {})
        for residue in residue_iter(atoms):
            orig_res_num = int(np.unique(residue.res_id)[0])
            for cdr_name, positions in chain_cdrs.items():
                if orig_res_num in positions:
                    cdr_residues.setdefault(f"{new_chain}:{cdr_name}", []).append(
                        residue_counter
                    )
            residue.res_id = np.full(len(residue), residue_counter)
            residue.ins_code = np.full(len(residue), "")
            atom_list.extend(residue)
            residue_counter += 1

    process_chain(heavy_chain, "H")
    process_chain(light_chain, "L")

    target_cdrs = cdr_definitions.get("T", {})
    for target in target_chains:
        chain_mask = structure.chain_id == target
        if not np.any(chain_mask):
            continue
        atoms = structure[chain_mask]
        atoms.chain_id = np.full(len(atoms), "T")
        for residue in residue_iter(atoms):
            orig_res_num = int(np.unique(residue.res_id)[0])
            for cdr_name, positions in target_cdrs.items():
                if orig_res_num in positions:
                    cdr_residues.setdefault(f"T:{cdr_name}", []).append(residue_counter)
            residue.res_id = np.full(len(residue), residue_counter)
            residue.ins_code = np.full(len(residue), "")
            atom_list.extend(residue)
            residue_counter += 1

    return array(atom_list), cdr_residues


def write_hlt_structure(
    structure_array: np.ndarray,
    cdr_residues: Dict[str, List[int]],
    output_path: str,
) -> None:
    """Write the converted structure and CDR annotations to disk."""

    pdb_file = PDBFile()
    pdb_file.set_structure(structure_array)

    with open(output_path, "w", encoding="utf-8") as handle:
        pdb_file.write(handle)
        for cdr, residues in sorted(cdr_residues.items()):
            for res_num in sorted(residues):
                handle.write(f"REMARK PDBinfo-LABEL: {res_num:4d} {cdr}\n")


def main() -> None:
    args = parse_args()

    cdr_definitions = parse_cdr_definitions(args.cdr)

    allowed_chains = set()
    if args.heavy:
        allowed_chains.add("H")
    if args.light:
        allowed_chains.add("L")
    if args.target:
        allowed_chains.add("T")

    invalid_chains = sorted(set(cdr_definitions) - allowed_chains)
    if invalid_chains:
        raise SystemExit(
            "CDR definitions were provided for chains without mappings: "
            + ", ".join(invalid_chains)
        )

    output_path = args.output or args.input_pdb.replace(".pdb", "_HLT.pdb")

    hlt_structure, cdr_residues = convert_to_hlt(
        input_pdb=args.input_pdb,
        heavy_chain=args.heavy,
        light_chain=args.light,
        target_chains=args.target,
        cdr_definitions=cdr_definitions,
    )

    write_hlt_structure(hlt_structure, cdr_residues, output_path)


if __name__ == "__main__":
    main()

