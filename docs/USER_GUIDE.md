# EasyClassifier – User Guide

EasyClassifier builds and evaluates classification models from your data
file by asking you simple questions. You do not write any code. This guide
takes you from an empty computer to your first report.

**Contents**

1. [What you need](#1-what-you-need)
2. [Install Python](#2-install-python)
3. [Install EasyClassifier](#3-install-easyclassifier)
4. [Try it with the demo (2 minutes)](#4-try-it-with-the-demo-2-minutes)
5. [Prepare your own data](#5-prepare-your-own-data)
6. [Run it on your data – what each step means](#6-run-it-on-your-data--what-each-step-means)
7. [Your results](#7-your-results)
8. [How long does it take?](#8-how-long-does-it-take)
9. [Optional extras](#9-optional-extras)
10. [Updating and uninstalling](#10-updating-and-uninstalling)
11. [Troubleshooting](#11-troubleshooting)
12. [Citing EasyClassifier](#12-citing-easyclassifier)

---

## 1. What you need

| | |
|---|---|
| **Computer** | Windows 10 or 11, macOS, or Linux |
| **Python** | Version **3.10 or newer**. 3.12, 3.13 or 3.14 are recommended (3.10 stops receiving security updates at the end of October 2026). https://www.python.org/downloads/|
| **Disk space** | About 400 MB for Python's scientific libraries |
| **Internet** | Only while installing |
| **Your data** | A `.csv` file or an Excel `.xlsx` file (see [section 5](#5-prepare-your-own-data)) |
| **Optional** | A LaTeX program, if you want the report as a PDF ([section 9](#9-optional-extras)) |

You will type a few commands into a **terminal** (a window where you type
instructions). Every command is given below exactly as you should type it.

* **Windows:** the terminal is called *Command Prompt*. Open it from the
  Start menu by typing `cmd` and pressing Enter.
* **macOS:** the terminal is called *Terminal*. Open it with Spotlight
  (⌘ + Space, type `Terminal`, press Return).
* **Linux:** open *Terminal* from your applications menu.

---

## 2. Install Python

Skip this section if Python 3.10 or newer is already installed. To check,
type in the terminal:

| Windows | macOS / Linux |
|---|---|
| `py --version` | `python3 --version` |

If you see `Python 3.10` or a higher number, go to [section 3](#3-install-easyclassifier).

### Windows

1. Install the **Python install manager**, the tool recommended by
   python.org for Windows. The easiest way is from the **Microsoft Store**:
   open the Store, search for *Python install manager*, and click
   *Get* / *Install*. (It can also be downloaded from
   [python.org/downloads/windows](https://www.python.org/downloads/windows/).)
2. Open **Command Prompt** and type:

   ```
   py --version
   ```

   The first time, the install manager may ask a few configuration
   questions; the suggested answers are fine. If you then see a line such
   as `Python 3.14.x`, Python is ready.
3. If instead you are told that no Python is installed, type the line
   below, then check again with `py --version`:

   ```
   py install 3.14
   ```

> **Already using the classic python.org installer?** That works too. During
> installation, tick **"Add python.exe to PATH"**. All the `py` commands in
> this guide work the same way.

### macOS

1. Go to [python.org/downloads/macos](https://www.python.org/downloads/macos/)
   and download the latest **macOS 64-bit universal2 installer** for Python
   3.13 or 3.14.
2. Open the downloaded file and follow the steps.
3. Open **Terminal** and check:

   ```
   python3 --version
   ```

### Linux

Python 3 is usually already installed. On Ubuntu or Debian you also need
the tool for separate environments:

```
sudo apt install python3-venv python3-pip
```

---

## 3. Install EasyClassifier

### Windows

In **Command Prompt**:

```
py -m pip install easyclassifier
```

### macOS

In **Terminal**:

```
python3 -m pip install easyclassifier
```

> If you installed Python with **Homebrew** and see the message
> *externally-managed-environment*, follow the Linux steps below instead.

### Linux (and Homebrew Python on macOS)

Recent Linux systems ask you to install programs like this one into a
separate *environment*. Type these three lines:

```
python3 -m venv ~/easyclassifier-env
~/easyclassifier-env/bin/python -m pip install easyclassifier
~/easyclassifier-env/bin/python -m easyclassifier --version
```

From now on, start EasyClassifier with:

```
~/easyclassifier-env/bin/python -m easyclassifier
```

### Check the installation

| Windows | macOS | Linux |
|---|---|---|
| `py -m easyclassifier --version` | `python3 -m easyclassifier --version` | `~/easyclassifier-env/bin/python -m easyclassifier --version` |

You should see something like:

```
EasyClassifier 0.8.2 (Python 3.13.2)
```

Installation downloads EasyClassifier and the scientific libraries it uses
(pandas, NumPy, scikit-learn, matplotlib, openpyxl). This takes one to a few
minutes.

---

## 4. Try it with the demo (2 minutes)

Start EasyClassifier:

| Windows | macOS | Linux |
|---|---|---|
| `py -m easyclassifier` | `python3 -m easyclassifier` | `~/easyclassifier-env/bin/python -m easyclassifier` |

> The shorter command `easyclassifier` also works on many computers. If it
> says *not recognized* or *command not found*, use the commands above;
> they always work.

Then:

1. Press **ENTER** to choose *Beginner* mode.
2. When asked for your data file, type **`demo`** and press ENTER.
3. Press **ENTER** at every remaining question to accept the suggested
   answers.

EasyClassifier analyses a small built-in dataset (150 Iris flowers, 3
species) with seven classifiers and finishes in well under a minute. At
the end it tells you where the results are:

```
[OK] All results were saved in this folder:
  C:\Users\you\Results\demo_iris_2026-09-28_10-15-02
...
Finished. Start with report.pdf: it explains the results in plain words
and includes a Methods paragraph you can adapt for a paper.
```

Open that folder and look at `report.pdf` (or `report.tex` if you have no
LaTeX, see [section 9](#9-optional-extras)).

**Useful keys at any question**

| Type | What it does |
|---|---|
| ENTER | accept the suggested answer (shown as *default* or *ENTER = …*) |
| a number | choose that option |
| `?` and a number, e.g. `?2` | explain option 2 in plain words |
| Ctrl + C | stop EasyClassifier |

---

## 5. Prepare your own data

EasyClassifier reads `.csv` files and Excel `.xlsx` files (the first sheet).
Your table should look like this:

| PatientAge | BloodPressure | Smoker | City | Diagnosis |
|---|---|---|---|---|
| 54 | 130 | Yes | Amman | Healthy |
| 61 | 145 | No | Irbid | Sick |
| 47 | | No | Zarqa | Healthy |

Checklist:

* **One row per case** (patient, student, customer, sample …).
* **The first row holds the column names.**
* **One column holds the groups you want to predict** (here *Diagnosis*).
  If it contains numbers such as a score or an income, EasyClassifier will
  offer to split them into groups (Low / Medium / High, or at a value you
  choose).
* **Empty cells are fine**; EasyClassifier fills them in or removes those
  rows – you choose.
* **Numbers must be plain numbers**: `72.5`, not `72.5 kg`. Decimal commas
  (`72,5`) are fine in files saved by Excel with semicolons.
* **No totals, notes, merged cells or charts** in the sheet – just the table.
* ID or name columns may stay in the file; EasyClassifier recognises and
  leaves them out, and tells you.
* Old Excel files (`.xls`) must first be saved as `.xlsx` or `.csv`
  (in Excel: *File → Save As*).

---

## 6. Run it on your data – what each step means

**Open the terminal in the folder where your data file is.** Results are
saved in a `Results` folder inside it.

* **Windows:** open the folder in File Explorer, click the address bar at
  the top, type `cmd`, press Enter.
* **macOS / Linux:** in Terminal, type `cd ` (with a space), drag the folder
  from Finder/Files into the Terminal window, press Return.

Then start EasyClassifier (see the table in [section 4](#4-try-it-with-the-demo-2-minutes)).
It shows *Step 1 of 6* to *Step 6 of 6*:

| Step | What you see | What to do |
|---|---|---|
| **Mode** | Beginner / Advanced / Research | *Beginner* asks only essential questions; *Advanced* lets you review each preparation step. |
| **1. Load your data** | "What is the name of your data file?" and the current folder | Type the file name (e.g. `survey.xlsx`), **or drag the file into the window**, then press ENTER. |
| **2. Data inspection** | Number of rows and columns, missing values, duplicates | Nothing – just read it. |
| **3. Choose what to predict** | Every column with a short description, e.g. `Diagnosis - 2 categories (Healthy, Sick)` | Press ENTER for the suggested column, or type its number. With many columns, type part of a name to search. If you pick an ID or a measurement, EasyClassifier explains and offers choices. |
| **4. Prepare the data** | What will be done with missing values, duplicates and text columns | Beginner: nothing. Advanced: choose each step (`?N` explains the options). |
| **5. Choose classifier(s)** | A list of methods | ENTER tries all of them (recommended). If you choose K-Nearest Neighbours, you also choose the distance – the Hassanat distance is suggested. |
| **6. Review your choices** | A summary | ENTER to start, or `n` to stop. |

While it runs you see *Training …* for each classifier. Then the results
appear on screen, for example:

```
Selected classifier: Support Vector Machine (SVM)

Final score on the untouched 20% test set - these are the numbers to report:
  * Accuracy           96.67%
  ...
Comparison of all classifiers (balanced accuracy, ranked) - used only to
choose the winner; slightly optimistic, do not report as the final result:
  * Support Vector Machine (SVM)        96.60%
  * Logistic Regression                 96.60%
  ...
```

**Which number should I report?** The **final score** – not the best score
in the comparison table. When several classifiers are compared, the winner
partly won by luck; the final score is measured on data that played no part
in choosing it (a 20 % test set kept aside from the start, or – for small
datasets – nested cross-validation).

---

## 7. Your results

Every run creates a **new folder**, so earlier results are never
overwritten:

```
Results/
  survey_2026-09-28_10-15-02/
    report.pdf              ← start here
    report.tex
    summary.csv
    results.xlsx
    predictions.csv
    feature_importance.csv
    trained_model.pkl
    citations.txt
    log.txt
    figures/
      class_distribution.png
      comparison.png           (when several classifiers were compared)
      confusion_matrix.png
      roc_curve.png
      feature_importance.png
      ...                      (more, depending on your data and choices)
```

| File | What it is |
|---|---|
| **report.pdf** | The full report in plain language: the question, the data, a **Methods paragraph you can adapt for a paper**, the results with an explanation of every number, the figures, which columns mattered, and the references to cite. |
| report.tex | The same report as a LaTeX file. Open it in Overleaf or any LaTeX editor to change the wording or layout. |
| summary.csv / results.xlsx | All scores in a table. The row starting with *FINAL* holds the numbers to report. |
| predictions.csv | For each tested row: the true class and the predicted class. |
| feature_importance.csv | How much each column helped the prediction. |
| trained_model.pkl | The trained model, including all preparation steps (for use by a programmer on new data). |
| citations.txt | What to cite if you publish. |
| log.txt | Every choice and step, with times – useful for your records and for reviewers. |
| figures/ | The figures as images (300 dpi, ready for journals), explained in the report. |

### The figures

By default you get the five figures most used in classification papers.
Two more are added automatically when your data need them. In *Advanced*
mode you can pick any of them, and three optional extras.

| Figure | When | What it shows |
|---|---|---|
| Class distribution | always | Rows in each class. |
| Classifier comparison | always, if several were compared | Each classifier's score in every test fold – is the winner clearly better? The report states this in words. |
| Confusion matrix | always | Which classes are confused, as counts and percentages. |
| ROC curves | always | How well each class is separated (one curve per class when there are 3 or more). |
| Feature importance | always | Which columns the model relies on. |
| Precision-recall curves | added when classes are unequal in size | The better view when one class is rare. |
| Missing-value map | added when the data have empty cells | Where the gaps are. |
| Correlation matrix | optional | Which numeric columns move together. |
| Columns by class | optional | How the most important columns differ between classes. |
| Learning curve | optional | Would collecting more data help? The report answers in words. Needs at least 20 rows. |

**Colours and file formats (Advanced mode).** Choose one of four themes:
*colour-blind safe* (the default, recommended for papers), *greyscale* (for
printed journals – classes are told apart by patterns and line styles),
*high contrast* (slides and posters), or *soft*. A class keeps the same
colour in every figure. Figures are saved as PNG at 300 dpi; you can also
ask for PDF or SVG copies, which stay sharp at any size (PDF is best for
journals; SVG can be edited in Inkscape or Illustrator).

---

## 8. How long does it take?

| Data | Classifiers | Typical time |
|---|---|---|
| Up to a few hundred rows | all | about a minute |
| 500 to 2,000 rows | all | a few minutes, sometimes more |
| 500 to 2,000 rows | Random Forest, Logistic Regression, Naive Bayes, Decision Tree | about a minute |
| More than 2,000 rows | all | usually quicker than just below 2,000 (see below) |
| More than 20,000 rows | SVM, KNN or Neural Network | can take a long time; EasyClassifier warns you first |

When several classifiers are compared, datasets of up to 2,000 rows use
nested cross-validation for the final score: it repeats the whole comparison
five times, which takes longer but gave the most accurate scores in our
benchmarks. Larger datasets use a final test set, which is accurate there
and much faster. Times depend on your computer.

---

## 9. Optional extras

### PDF report (LaTeX)

`report.pdf` is made from `report.tex` by a LaTeX program. Without one,
EasyClassifier still writes `report.tex` and tells you. To get PDFs:

* **Windows:** install [MiKTeX](https://miktex.org/). When the installer
  asks what to do about missing packages, choose to install them
  automatically (*Always* / *Yes*). Otherwise MiKTeX opens a question window
  in the middle of making the report and the PDF step waits for it.
* **macOS:** install [MacTeX](https://www.tug.org/mactex/).
* **Linux (Ubuntu/Debian):**
  `sudo apt install texlive-latex-recommended texlive-latex-extra lmodern`

**Or, without installing anything:** make a zip file of the results folder,
go to [overleaf.com](https://www.overleaf.com), choose *New project → Upload
project*, upload the zip, and open `report.tex`. Overleaf shows the PDF.

Column names in non-Latin scripts (for example Arabic) appear as `?` in the
PDF. You can rename those columns in your data file, or edit `report.tex`.

### More classifiers (XGBoost and LightGBM)

| Windows | macOS |
|---|---|
| `py -m pip install "easyclassifier[full]"` | `python3 -m pip install "easyclassifier[full]"` |

They then appear in the classifier list.

---

## 10. Updating and uninstalling

| | Windows | macOS |
|---|---|---|
| Update | `py -m pip install --upgrade easyclassifier` | `python3 -m pip install --upgrade easyclassifier` |
| Uninstall | `py -m pip uninstall easyclassifier` | `python3 -m pip uninstall easyclassifier` |

On Linux use `~/easyclassifier-env/bin/python -m pip …`, or simply delete
the `~/easyclassifier-env` folder to remove everything.

Your `Results` folders are never deleted by updating or uninstalling.

---

## 11. Troubleshooting

| Message or problem | What to do |
|---|---|
| `'easyclassifier' is not recognized…` / `command not found` | Use `py -m easyclassifier` (Windows) or `python3 -m easyclassifier` (macOS). |
| `'py' is not recognized` (Windows) | Python is not installed, or the classic installer was used without *Add python.exe to PATH*. Repeat [section 2](#2-install-python). |
| `python3: command not found` (macOS) | Install Python from python.org ([section 2](#2-install-python)). |
| `No module named easyclassifier` | It was installed for a different Python. Install and start it with the **same** first word (`py`, `python3`, or the environment path). |
| `externally-managed-environment` | Use the separate-environment steps in [section 3](#linux-and-homebrew-python-on-macos). |
| Installation fails with a very long file path (Windows) | Windows limits path length. Enable long paths (search the web for "Windows enable long paths", or ask your IT support) and install again. |
| `The file cannot be found` | Drag the file into the window instead of typing its name, or check the *Current folder* shown in Step 1. |
| A column of numbers is described as *categories* or *text values* | The column contains non-numbers such as units (`72 kg`) or notes (`n/a`). Clean the column in Excel and save again. |
| `No LaTeX program was found` | See [section 9](#9-optional-extras); `report.tex` has already been written. |
| `LaTeX … could not build the PDF` | Details are in `report_latex_errors.log`. Uploading to Overleaf usually works. |
| It is very slow | Large data with SVM, KNN or Neural Network. Press Ctrl + C and choose faster classifiers ([section 8](#8-how-long-does-it-take)). |
| Anything else | Every step is in `log.txt`. Please report problems on the project page, attaching `log.txt` (not your data). |

---

## 12. Citing EasyClassifier

If you publish results obtained with EasyClassifier, please cite it. The
exact references for your analysis are in `citations.txt` and at the end of
`report.pdf`. They include the software, scikit-learn, NumPy, pandas and
Matplotlib, a reference for every method that was used (each classifier, the
KNN distance, the validation design, the measures and curves that need a
source), and – when K-nearest neighbours with the Hassanat distance was
used –

> Hassanat, A. B., Alkafaween, E., Tarawneh, A. S., & Elmougy, S. (2022).
> Applications review of Hassanat distance metric. In *2022 International
> Conference on Emerging Trends in Computing and Engineering Applications
> (ETCEA)*, Karak, Jordan (pp. 1–6). IEEE.
> https://doi.org/10.1109/ETCEA57049.2022.10009844
>
> Hassanat, A. B. (2014). Dimensionality invariant similarity measure.
> arXiv preprint arXiv:1409.0923. https://arxiv.org/abs/1409.0923

To cite the software itself: Hassanat, A. B. A. (2026). EasyClassifier (version 0.8.2). Zenodo. https://doi.org/10.5281/zenodo.23122901
