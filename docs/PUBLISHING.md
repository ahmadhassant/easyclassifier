# Publishing EasyClassifier – step by step

This is the checklist for putting EasyClassifier online: on GitHub (the
code), on PyPI (so that `pip install easyclassifier` works) and on Zenodo
(a permanent DOI for citing it). Everything in the project folder is already
prepared; these steps need your own accounts, so only you can do them.

Time needed: about one hour, once.

---

## 1. GitHub: put the code online

1. Sign in at [github.com](https://github.com) as **ahmadhassant**.
2. Click **+ → New repository**.
   * Repository name: `easyclassifier`
   * Description: *Machine learning classification without programming*
   * **Public**
   * Do **not** tick "Add a README", ".gitignore" or "license" – they are
     already in the project.
   * Click **Create repository**.
3. Open a terminal in the project folder (`C:\research C\Easy Classifier`)
   and type:

   ```
   git push -u origin main
   ```

   (The project is already a git repository with its first commit, and
   `origin` already points to github.com/ahmadhassant/easyclassifier.)
   Git asks you to sign in to GitHub the first time; a browser window opens
   for that.

   If `git` is not installed: download it from
   [git-scm.com](https://git-scm.com/download/win) and install with the
   default options, then repeat step 3.

4. On the repository page, open the **Actions** tab. The *tests* workflow
   starts automatically and tests EasyClassifier on Windows, macOS and
   Linux with Python 3.10–3.14. After some minutes every job should show a
   green tick. (A red cross means something needs fixing on that system –
   the log shows what.)

5. Optional but recommended: in **Settings → General → Features**, keep
   *Issues* switched on, so users can report problems.

## 2. Zenodo: a permanent DOI

Do this **before** the first release, so the release gets a DOI.

1. Sign in at [zenodo.org](https://zenodo.org) with your GitHub account
   (*Log in → GitHub*). Your ORCID can be linked in the Zenodo profile.
2. Go to **GitHub** in the Zenodo menu (your profile → *GitHub*), find
   `ahmadhassant/easyclassifier` and switch it **On**.

From now on, every GitHub release is archived by Zenodo with its own DOI,
using the information in `.zenodo.json`.

## 3. PyPI: `pip install easyclassifier`

1. Create an account at [pypi.org](https://pypi.org/account/register/) and
   switch on two-factor authentication (PyPI requires it).
2. Go to **Your account → Publishing → Add a new pending publisher** and
   fill in:
   * PyPI project name: `easyclassifier`
   * Owner: `ahmadhassant`
   * Repository name: `easyclassifier`
   * Workflow name: `release.yml`
   * Environment name: `pypi`
3. On GitHub: **Settings → Environments → New environment**, name it
   `pypi`, click *Configure environment*, and save.

This is "trusted publishing": GitHub proves to PyPI that the upload comes
from your repository, so no password or token is stored anywhere.

(Optional test first: repeat step 2 on [test.pypi.org](https://test.pypi.org)
if you want a dry run.)

## 4. The first release

1. On GitHub: **Releases → Draft a new release**.
2. *Choose a tag*: type `v0.8.1` and click *Create new tag*.
3. Title: `EasyClassifier 0.8.1`. In the description, paste the 0.8.1 part
   of `CHANGELOG.md`.
4. Click **Publish release**.

What happens automatically:

* the *release* workflow builds the package and uploads it to PyPI – after
  a few minutes `pip install easyclassifier` works for everyone;
* Zenodo archives the release and gives it a DOI (shown on your Zenodo
  "Uploads" page).

## 5. After the first release

1. Copy the DOI badge from Zenodo (click the DOI on the record page, copy
   the Markdown) and paste it at the top of `README.md`, next to the other
   badges. Add the DOI to `CITATION.cff` as

   ```
   doi: 10.5281/zenodo.NNNNNNN
   ```

2. Commit and push:

   ```
   git add README.md CITATION.cff
   git commit -m "Add Zenodo DOI"
   git push
   ```

3. Check that installing works on a clean computer:
   `py -m pip install easyclassifier` then `py -m easyclassifier --version`.

## Later releases

1. Change the version number in `pyproject.toml` **and**
   `src/easyclassifier/__init__.py` (and `version` / `date-released` in
   `CITATION.cff`).
2. Add a section to `CHANGELOG.md`.
3. Commit, push, and draft a new GitHub release with the new tag
   (e.g. `v0.9.0`). PyPI and Zenodo update automatically.

PyPI never accepts the same version number twice, so always increase it.

## For JOSS (Journal of Open Source Software)

JOSS asks for at least six months of public development history (releases,
issues, changes) before submission. Pushing to GitHub now starts that clock.
A draft `paper.md` for JOSS can be prepared later from the arXiv paper.
