"""Common I/O utilities for loading and saving analysis data.

This module provides reusable functions for loading and saving common data formats
used in analysis pipelines (CSV, JSON, NumPy arrays, etc.).
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


# === CSV Operations ===

def load_csv(
    path: Union[str, Path],
    index_col: Optional[Union[int, str]] = None,
    **kwargs
) -> pd.DataFrame:
    """Load a CSV file as a pandas DataFrame.
    
    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    index_col : int, str, or None, optional
        Column to use as the row labels of the DataFrame.
    **kwargs
        Additional arguments passed to pd.read_csv.
    
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    
    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    logger.debug(f"Loading CSV from {path}")
    return pd.read_csv(path, index_col=index_col, **kwargs)


def save_csv(
    df: pd.DataFrame,
    path: Union[str, Path],
    index: bool = False,
    **kwargs
) -> None:
    """Save a DataFrame to a CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : str or Path
        Output path for the CSV file.
    index : bool, default=False
        Whether to write row names (index).
    **kwargs
        Additional arguments passed to pd.DataFrame.to_csv.
    
    Returns
    -------
    None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Saving CSV to {path}")
    df.to_csv(path, index=index, **kwargs)


# === JSON Operations ===

def load_json(path: Union[str, Path]) -> Any:
    """Load data from a JSON file.
    
    Parameters
    ----------
    path : str or Path
        Path to the JSON file.
    
    Returns
    -------
    Any
        Parsed JSON data (typically dict or list).
    
    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    logger.debug(f"Loading JSON from {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(
    data: Any,
    path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False,
    **kwargs
) -> None:
    """Save data to a JSON file.
    
    Parameters
    ----------
    data : Any
        Data to serialize (typically dict or list).
    path : str or Path
        Output path for the JSON file.
    indent : int, default=2
        Number of spaces for indentation.
    ensure_ascii : bool, default=False
        If True, escape non-ASCII characters.
    **kwargs
        Additional arguments passed to json.dump.
    
    Returns
    -------
    None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Saving JSON to {path}")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


# === NumPy Operations ===

def load_numpy(path: Union[str, Path], allow_pickle: bool = True) -> np.ndarray:
    """Load a NumPy array from a .npy file.
    
    Parameters
    ----------
    path : str or Path
        Path to the .npy file.
    allow_pickle : bool, default=True
        Allow loading pickled object arrays.
    
    Returns
    -------
    np.ndarray
        Loaded array.
    
    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NumPy file not found: {path}")
    
    logger.debug(f"Loading NumPy array from {path}")
    return np.load(path, allow_pickle=allow_pickle)


def save_numpy(
    arr: np.ndarray,
    path: Union[str, Path],
    compressed: bool = False
) -> None:
    """Save a NumPy array to a .npy or .npz file.
    
    Parameters
    ----------
    arr : np.ndarray
        Array to save.
    path : str or Path
        Output path for the .npy file.
    compressed : bool, default=False
        If True, save as compressed .npz format.
    
    Returns
    -------
    None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Saving NumPy array to {path}")
    if compressed:
        np.savez_compressed(path, arr)
    else:
        np.save(path, arr)


def load_npz(path: Union[str, Path], allow_pickle: bool = True) -> Dict[str, np.ndarray]:
    """Load a compressed NumPy .npz file as a dictionary.
    
    Parameters
    ----------
    path : str or Path
        Path to the .npz file.
    allow_pickle : bool, default=True
        Allow loading pickled object arrays.
    
    Returns
    -------
    dict
        Dictionary mapping array names to NumPy arrays.
    
    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")
    
    logger.debug(f"Loading NPZ from {path}")
    z = np.load(path, allow_pickle=allow_pickle)
    return dict(z)


def save_npz(data: Dict[str, np.ndarray], path: Union[str, Path]) -> None:
    """Save multiple NumPy arrays to a compressed .npz file.
    
    Parameters
    ----------
    data : dict
        Dictionary mapping array names to NumPy arrays.
    path : str or Path
        Output path for the .npz file.
    
    Returns
    -------
    None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Saving NPZ to {path}")
    np.savez_compressed(path, **data)


# === Combined Load/Save for Distance Matrices ===

def load_distance_data(
    matrix_path: Union[str, Path],
    subjects_path: Optional[Union[str, Path]] = None
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Load distance matrix and optional subject list.
    
    Parameters
    ----------
    matrix_path : str or Path
        Path to distance matrix (.npy or .csv).
    subjects_path : str or Path, optional
        Path to subjects JSON file. If None, only matrix is returned.
    
    Returns
    -------
    matrix : np.ndarray
        Distance matrix.
    subjects : list of str or None
        Subject IDs if subjects_path provided, else None.
    """
    matrix_path = Path(matrix_path)
    
    # Load matrix
    if matrix_path.suffix == '.npy':
        matrix = load_numpy(matrix_path)
    elif matrix_path.suffix == '.csv':
        df = load_csv(matrix_path, index_col=0)
        matrix = df.values
    else:
        raise ValueError(f"Unsupported matrix format: {matrix_path.suffix}")
    
    # Load subjects if path provided
    subjects = None
    if subjects_path is not None:
        subjects = load_json(subjects_path)
    
    return matrix, subjects


def save_distance_data(
    matrix: np.ndarray,
    matrix_path: Union[str, Path],
    subjects: Optional[List[str]] = None,
    subjects_path: Optional[Union[str, Path]] = None
) -> None:
    """Save distance matrix and optional subject list.
    
    Parameters
    ----------
    matrix : np.ndarray
        Distance matrix to save.
    matrix_path : str or Path
        Output path for matrix (.npy).
    subjects : list of str, optional
        Subject IDs to save.
    subjects_path : str or Path, optional
        Output path for subjects JSON. Required if subjects provided.
    
    Returns
    -------
    None
    """
    save_numpy(matrix, matrix_path)
    
    if subjects is not None:
        if subjects_path is None:
            raise ValueError("subjects_path required when subjects provided")
        save_json(subjects, subjects_path)
