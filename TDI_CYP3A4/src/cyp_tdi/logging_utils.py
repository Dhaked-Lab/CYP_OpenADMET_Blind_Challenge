import logging, sys
from pathlib import Path


def get_logger(name="cyp_tdi", logfile: Path | None = None) -> logging.Logger:
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in lg.handlers):
        h = logging.StreamHandler(sys.stdout); h.setFormatter(fmt); lg.addHandler(h)
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(Path(logfile).resolve()) for h in lg.handlers):
            fh = logging.FileHandler(logfile, encoding="utf-8"); fh.setFormatter(fmt); lg.addHandler(fh)
    return lg
