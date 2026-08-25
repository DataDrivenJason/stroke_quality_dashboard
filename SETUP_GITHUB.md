# Getting this onto GitHub and live

Written for Windows, since that is what you are on. Everything below runs in
**Anaconda Prompt** or **PowerShell** — or use the VS Code Source Control panel
if you prefer clicking.

The repository is already initialised with a first commit, so you are picking
up from step 2.

---

## 1. First, replace the placeholder in the README

The README badges and the demo link contain `USERNAME`. Swap in your GitHub
username. In PowerShell, from the project folder:

```powershell
(Get-Content README.md) -replace 'USERNAME','your-github-username' | Set-Content README.md
git commit -am "Point README badges and demo link at the real repo"
```

If you choose a repository name other than `stroke-quality-dashboard`, replace
that in the README too.

---

## 2. Create the repository on GitHub

**In the browser:** go to [github.com/new](https://github.com/new), name it
`stroke-quality-dashboard`, set it **Public**, and — importantly — do **not**
tick "Add a README", "Add .gitignore" or "Choose a license". This project
already has them, and pre-adding files creates a conflict you then have to
merge on your first push.

**Or with the GitHub CLI**, if you have it (`winget install GitHub.cli`):

```powershell
gh auth login
gh repo create stroke-quality-dashboard --public --source=. --remote=origin --push
```

That does step 3 as well, and you can skip ahead.

---

## 3. Push

```powershell
git remote add origin https://github.com/your-github-username/stroke-quality-dashboard.git
git branch -M main
git push -u origin main
```

If you have not used git on this machine before, set your identity first —
otherwise the commit is attributed to a placeholder:

```powershell
git config --global user.name "Jason McGrath"
git config --global user.email "your@email.com"
```

Then re-run the existing commit with the right author:

```powershell
git commit --amend --reset-author --no-edit
```

GitHub will ask you to authenticate on the first push. The browser flow is the
easiest route; a personal access token also works.

---

## 4. Check CI went green

Open the **Actions** tab on your new repo. The `tests` workflow runs the full
suite on four Python/pandas combinations plus a check that the app actually
boots. It takes roughly three minutes. Once it passes, the badge at the top of
the README turns green.

If a job fails, the log tells you which combination and which test. Nothing in
the workflow needs secrets or configuration.

---

## 5. Deploy the live app

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Authorise Streamlit to see your repositories.
3. **Create app** → **Deploy a public app from GitHub**.
4. Repository `your-github-username/stroke-quality-dashboard`, branch `main`,
   main file path `app.py`.
5. Under **Advanced settings**, set Python version to **3.12**.
6. Deploy.

The first build takes two or three minutes while it installs dependencies. You
get a URL like `https://your-github-username-stroke-quality-dashboard.streamlit.app`,
which you can customise in the app's settings.

Every push to `main` redeploys automatically.

### Things worth knowing about Community Cloud

- **It sleeps.** Apps with no traffic for about a week go to sleep and take
  roughly 30 seconds to wake on the next visit. Fine for a portfolio link,
  irritating for a live operational tool.
- **Resource limits.** The free tier gives about 1 GB of RAM. This app peaks
  around 400 MB generating the demonstration data, so there is headroom, but a
  much larger real extract would not fit.
- **No authentication.** Anyone with the URL can open it. That is fine for
  simulated data and **not** fine for anything real — see below.
- **The `data/` folder is ephemeral.** The container regenerates the
  demonstration data on first load, which is exactly what you want here.

---

## 6. Before you ever point it at real data

Three things change:

**Do not deploy it publicly.** Community Cloud has no authentication. A real
extract needs to run inside your organisation's infrastructure, behind its
existing access controls.

**Trust the .gitignore, but check anyway.** It blocks `.csv`, `.parquet`,
`.xlsx`, `.sav`, `.dta`, `.sqlite` and more outright. Before any commit that
touches data handling, run:

```powershell
git status --short
git diff --cached --stat
```

The failure mode here is unrecoverable. A force-push does not remove a blob
from a fork, a clone, or GitHub's cache, so the only reliable protection is not
committing it in the first place.

**Talk to your DPO first.** Even fully de-identified stroke audit extracts are
usually covered by a data-sharing or information-governance agreement that says
where the data may be processed. A public cloud service is a processing
location.

---

## 7. A decision you have not made yet: the licence

The repository currently has **no licence file**. Under GitHub's terms that
means default copyright — people can view and fork it, but have no right to
use, modify or redistribute it. That is a legitimate choice, and it is also
what happens by accident when nobody decides.

If you want it usable as a portfolio piece that others can build on, the two
usual options:

- **MIT** — anyone can do anything with it, including commercially, as long as
  they keep the copyright notice. The default for tooling and portfolio repos.
- **AGPL-3.0** — anyone can use and modify it, but if they run a modified
  version as a network service they must publish their changes. Sometimes
  preferred for health tooling, on the reasoning that improvements to a clinical
  tool should come back to the commons.

Either is one file: GitHub can add it for you via **Add file → Create new
file** → type `LICENSE` → "Choose a license template".

Also worth a thought before making it public: whether your employer has a
policy on publishing work-adjacent code. The code here contains no real data
and no organisational detail — the site names are invented — but the question
is usually about the work, not the data.
