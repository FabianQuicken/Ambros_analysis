from pathlib import Path
import joblib


def save_analysis(results: dict, filepath: str, compress: int = 3):
    """
    Save an analysis dictionary to disk using joblib.

    Parameters
    ----------
    results : dict
        Dictionary returned by the analysis function.
    filepath : str
        Target file path (recommended extension: .joblib).
    compress : int, optional
        Compression level (0–9). Default is 3 (good balance of speed/size).

    Returns
    -------
    Path
        Path to the saved file.
    """

    path = Path(filepath)

    # create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(results, path, compress=compress)

    return path


def load_analysis(filepath: str) -> dict:
    """
    Load a previously saved analysis dictionary.

    Parameters
    ----------
    filepath : str
        Path to the saved joblib file.

    Returns
    -------
    dict
        The analysis dictionary.
    """

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    results = joblib.load(path)

    return results