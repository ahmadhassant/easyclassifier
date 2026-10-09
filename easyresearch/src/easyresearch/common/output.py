"""Small helpers shared by the cores: number formatting, output folders,
LaTeX compilation, and the record of settings and software versions that
every run folder contains."""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import platform
import sys
import subprocess
from typing import Dict, Optional, Tuple

from easyclassifier.latex_report import find_engine


def fmt(v: float) -> str:
    """A number with sensible precision for any unit."""
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


def results_folder(out_root: str, source_stem: str) -> str:
    """A new folder for every run: <out_root>/Results/<data>_<date>_<time>."""
    stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                   for ch in source_stem)[:40] or "data"
    out = os.path.join(out_root, "Results", f"{safe}_{stamp}")
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    return out


def compile_tex(tex_text: str, out_dir: str,
                timeout: int = 180) -> Tuple[str, Optional[str], str]:
    """Write report.tex and compile it to report.pdf when LaTeX is found."""
    tex_path = os.path.join(out_dir, "report.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(tex_text)
    engine = find_engine()
    if engine is None:
        return tex_path, None, (
            "No LaTeX program was found, so only report.tex was written. "
            "Upload it with the 'figures' folder to an online editor such as "
            "Overleaf, or install MiKTeX to get report.pdf.")
    pdf_path = os.path.join(out_dir, "report.pdf")
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    runs = ([["tectonic", "report.tex"]] if engine == "tectonic" else
            [[engine, "-interaction=nonstopmode", "-halt-on-error",
              "report.tex"]] * 2)
    proc = None
    try:
        for run in runs:
            proc = subprocess.run(run, cwd=out_dir, capture_output=True,
                                  timeout=timeout)
            if proc.returncode != 0:
                break
    except (subprocess.TimeoutExpired, OSError) as exc:
        return tex_path, None, f"LaTeX ({engine}) did not finish: {exc}."
    finally:
        for ext in (".aux", ".out", ".toc"):
            p = os.path.join(out_dir, "report" + ext)
            if os.path.exists(p):
                os.remove(p)
    log = os.path.join(out_dir, "report.log")
    if proc is not None and proc.returncode == 0 and os.path.exists(pdf_path):
        if os.path.exists(log):
            os.remove(log)
        return tex_path, pdf_path, f"PDF created with {engine}."
    if os.path.exists(log):
        os.replace(log, os.path.join(out_dir, "report_latex_errors.log"))
    return tex_path, None, (f"LaTeX ({engine}) could not build the PDF; see "
                            "report_latex_errors.log. report.tex can still be "
                            "opened in Overleaf.")


VERSION_PACKAGES = ("easyresearch", "easyclassifier", "numpy", "pandas",
                    "scikit-learn", "scipy", "matplotlib", "joblib",
                    "statsmodels", "torch", "openpyxl", "xgboost", "lightgbm")


def software_versions() -> Dict[str, str]:
    """Python, operating system and the installed analysis packages."""
    from importlib.metadata import PackageNotFoundError, version
    found = {"Python": platform.python_version(),
             "Operating system": f"{platform.system()} {platform.release()} "
                                 f"({platform.machine()})"}
    for name in VERSION_PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            pass
    return found


def write_run_record(out_dir: str, task: str, data: str, column: str,
                     settings, resolved: Dict) -> None:
    """settings.json (what was asked and what was used) and versions.txt
    (the software that produced the results), so a run can be repeated and
    reported exactly."""
    asked = (dataclasses.asdict(settings) if dataclasses.is_dataclass(settings)
             else dict(settings))
    record = {"task": task, "data": data, "column": column,
              "settings": asked, "used": resolved,
              "created": _dt.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(out_dir, "settings.json"), "w",
              encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2, default=str)
    lines = [f"{k}: {v}" for k, v in software_versions().items()]
    with open(os.path.join(out_dir, "versions.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("Software used for these results\n\n" + "\n".join(lines)
                 + f"\n\nPython executable: {sys.executable}\n")


def figure_style(settings, known) -> Tuple[dict, Optional[list]]:
    """The figure theme and file formats chosen in the settings (as keyword
    arguments for the figure maker), and the figures to draw (None = the
    task's usual set). Unknown choices are refused in plain words."""
    from easyclassifier.figures import FORMATS, THEMES
    theme = getattr(settings, "figure_theme", "colorblind")
    formats = getattr(settings, "figure_format", "png")
    keys = getattr(settings, "figure_keys", None)
    if theme not in THEMES:
        raise ValueError(f"Unknown figure theme '{theme}'; choose one of: "
                         + ", ".join(THEMES) + ".")
    if formats not in FORMATS:
        raise ValueError(f"Unknown figure format '{formats}'; choose one of: "
                         + ", ".join(FORMATS) + ".")
    if keys is not None:
        unknown = [k for k in keys if k not in known]
        if unknown:
            raise ValueError("Unknown figure(s): " + ", ".join(unknown) + ".")
        keys = [k for k in known if k in keys]          # report order
    return dict(theme=theme, formats=formats), keys


def save_model(obj, path: str) -> Optional[str]:
    """Save the trained model; on failure, remove the partial file and return
    a plain message instead of stopping the analysis (all other results are
    still saved)."""
    import joblib
    try:
        joblib.dump(obj, path)
        return None
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(path):
            os.remove(path)
        return (f"The trained model could not be saved ({type(exc).__name__}: "
                f"{exc}); all other results were saved.")
