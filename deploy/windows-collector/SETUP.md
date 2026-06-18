# AERIS Windows Collector — Setup

Turns the Acer (i7-4710HQ / 16 GB / Win 10 Education) into an unattended data
collector that runs all four AERIS collectors hourly into a local SQLite
database. Built to survive a multi-week travel gap with no one at the keyboard:
code is deployed by `git pull`, and an independent liveness alarm pings a
dead-man's switch so a silent stall reaches your phone.

**Total time: ~90 minutes.** Do the steps in order. Boxes to tick are marked `[ ]`.

This box does **collection only**. It does *not* run Ollama / Llama 3 8B — the
GTX 860M has 2 GB VRAM and cannot fit an 8B model. Run the LLM phase on your Mac
later, as the Month 2 plan's risk table already assumes.

---

## What gets installed

- **Miniforge** — a minimal conda. Brings Python 3.11 and, critically, the
  `pygrib` + `eccodes` binaries that the NOAA GFS collector needs and that
  plain `pip` cannot install on Windows.
- **A conda env `aeris`** — the binary scientific packages.
- **Pip packages** — the pure-Python web/db/config stack (`requirements-collector.txt`).
- **A git checkout** — the code is a clone of `origin/main`; updates land via
  `deploy.bat` (`git fetch` + hard reset), never a manual copy.
- **Two scheduled tasks** — `run_collectors.bat` every hour, and
  `run_liveness.bat` every 30 min as a watchdog.

Database is **SQLite** at `C:\aeris-data\aeris.db` — one portable file, no
service to crash, and **outside the code tree** so no deploy can touch it. Copy
that file to your Mac when you're back to run detection.

`server\.env` (your API keys) is gitignored, so it never comes down with the
code and is never overwritten by a pull — it lives only on this box.

---

## Step 0 — Get the code onto the Acer via git (~10 min)

The code is deployed from GitHub so future updates are one command. `server\.env`
is **not** in git (it holds your keys), so you create it by hand in Step 3.

1. [ ] Install git. Either the Git for Windows installer (bundles a credential
       manager), or from the Miniforge Prompt once Miniforge is in (Step 1):
       ```bat
       conda install -c conda-forge git
       ```
2. [ ] Cache your GitHub credential so the unattended box never blocks on a prompt:
       ```bat
       git config --global credential.helper store
       ```
       (Git for Windows users can use `manager` instead, which encrypts it.)
3. [ ] Clone into `C:\temp\aeris\aeris`:
       ```bat
       git clone https://github.com/mason-cao/aeris.git C:\temp\aeris\aeris
       ```
       When prompted, the **password is a fine-grained Personal Access Token**
       (GitHub → Settings → Developer settings → Fine-grained tokens → repo
       `aeris`, **Contents: Read-only**) — *not* your account password.
4. [ ] Confirm the layout: `C:\temp\aeris\aeris\server` and
       `C:\temp\aeris\aeris\deploy` both exist.

> **Already deployed via USB/zip?** Convert that folder to a git checkout in
> place instead of re-cloning — it keeps your existing `server\.env`:
> ```bat
> copy C:\temp\aeris\aeris\server\.env C:\aeris-data\.env.bak
> cd /d C:\temp\aeris\aeris
> git init
> git remote remove origin 2>nul
> git remote add origin https://github.com/mason-cao/aeris.git
> git fetch origin
> git reset --hard origin/main
> ```
> `git reset --hard` only rewrites tracked files; `.env` is gitignored and the
> DB is external, so both survive. If `.env` ever vanishes, restore it from the
> `.bak`.

---

## Step 1 — Install Miniforge (~10 min)

1. [ ] Download `Miniforge3-Windows-x86_64.exe` from
       <https://github.com/conda-forge/miniforge/releases/latest>
2. [ ] Run it. Accept defaults — it installs to `%USERPROFILE%\miniforge3`.
3. [ ] Open **Miniforge Prompt** from the Start menu (use this for Steps 2 & 4,
       not the regular Command Prompt).

---

## Step 2 — Build the Python environment (~20 min)

In the Miniforge Prompt:

```bat
conda env create -f C:\temp\aeris\aeris\deploy\windows-collector\environment.yml
conda activate aeris
pip install -r C:\temp\aeris\aeris\deploy\windows-collector\requirements-collector.txt
```

1. [ ] `conda env create` finishes without error (this installs `pygrib` +
       `eccodes` — the part that would fail under plain pip).
2. [ ] `pip install` finishes without error.

If `conda env create` fails on `pygrib`, see **Troubleshooting** — the other
three collectors still work and you can move on.

---

## Step 3 — Create and configure `server\.env` (~5 min)

A fresh clone has no `.env`. Make one from the template:

```bat
copy C:\temp\aeris\aeris\server\.env.example  C:\temp\aeris\aeris\server\.env
```

Edit **`C:\temp\aeris\aeris\server\.env`** in Notepad:

1. [ ] Fill in your real API keys: `OPENAQ_API_KEY`, `OPENWEATHER_API_KEY`,
       `NASA_EARTHDATA_TOKEN`, `CDSE_USERNAME`, `CDSE_PASSWORD`.
2. [ ] Set these two lines exactly:
       ```
       DATABASE_URL=sqlite+aiosqlite:///C:/aeris-data/aeris.db
       AERIS_ENV=production
       ```
       - `sqlite+aiosqlite` — async SQLite driver, already installed.
       - Forward slashes are correct even on Windows.
       - The absolute path keeps the DB outside the repo, so deploys never touch it.
       - `AERIS_ENV=production` silences SQLAlchemy's per-query echo so the logs stay small.

---

## Step 4 — Create the database (~3 min)

In the Miniforge Prompt (env still active):

```bat
mkdir C:\aeris-data
cd C:\temp\aeris\aeris
python deploy\windows-collector\init_db.py
```

1. [ ] Output ends with `Tables ready in sqlite+aiosqlite:///C:/aeris-data/aeris.db`.

---

## Step 5 — Test a real collection run (~5 min)

```bat
cd C:\temp\aeris\aeris\server
python -m app.collectors.run_all
```

1. [ ] You see four result lines (`openaq`, `openweather`, `noaa_gfs`,
       `sentinel5p`). At least three should say `ok`.
2. [ ] **Run it a second time.** Every line should still say `ok` — this
       confirms duplicate rows are de-duplicated, not errored, so hourly runs
       are safe.

If `noaa_gfs` says `failed` because `pygrib` didn't install, that's acceptable —
see Troubleshooting. Do not block on it.

---

## Step 6 — Stop the laptop from sleeping (~5 min)

Open **Command Prompt as Administrator** and run:

```bat
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
powercfg /change hibernate-timeout-ac 0
powercfg /change hibernate-timeout-dc 0
powercfg /change disk-timeout-ac 0
powercfg /hibernate off
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
powercfg /setdcvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
powercfg /setactive SCHEME_CURRENT
```

This disables sleep/hibernate and makes **closing the lid do nothing** so the
laptop keeps running closed.

1. [ ] All commands ran without error.
2. [ ] **Leave the Acer plugged into AC power** for the whole trip.
3. [ ] Prefer an **Ethernet cable** over Wi-Fi for a stable weeks-long connection.
       On Wi-Fi, confirm the network is set to "Connect automatically".

---

## Step 7 — Pause Windows Update (~3 min)

A forced update reboot mid-window would interrupt collection.

1. [ ] **Settings → Update & Security → Windows Update → Pause updates for 7 days.**
       Click it again if it lets you extend further.

The scheduled tasks survive a reboot anyway (Step 8), so a reboot costs at most
one hour of data — but pausing avoids it entirely.

---

## Step 8 — Schedule the hourly collector task (~5 min)

In the **Administrator Command Prompt**:

```bat
schtasks /create /tn "AERIS Collector" /tr "C:\temp\aeris\aeris\deploy\windows-collector\run_collectors.bat" /sc hourly /rl HIGHEST /f
```

1. [ ] Output says `SUCCESS: The scheduled task "AERIS Collector" ...`.

This runs every hour **while you are logged in**. For the trip, that's fine —
just **leave the Acer logged in** (don't sign out; the lid can be closed).

*Optional hardening* — to run even through a reboot to the login screen, recreate
the task with your account + password (only works if your account has a password):

```bat
schtasks /create /tn "AERIS Collector" /tr "C:\temp\aeris\aeris\deploy\windows-collector\run_collectors.bat" /sc hourly /rl HIGHEST /ru "%USERNAME%" /rp * /f
```

It will prompt for your Windows password.

---

## Step 9 — Verify the collector task works (~5 min)

```bat
schtasks /run /tn "AERIS Collector"
```

Wait ~30 seconds, then open the log:

```bat
notepad "C:\aeris-data\logs\collector.log"
```

1. [ ] The log shows a `===== RUN ... =====` block, four collector lines, and
       `exit_code=0` (or `exit_code=1` only if GFS is the lone failure).

The collector now fires at the top of every hour. Next, set up updates and the
watchdog.

---

## Step 10 — Updating the code later with `deploy.bat`

Once running, you never copy files again. To ship a change: push it from the Mac,
then on the Acer run (from the **Miniforge Prompt**, which has git + conda on PATH):

```bat
C:\temp\aeris\aeris\deploy\windows-collector\deploy.bat
```

It does `git fetch` → `git reset --hard origin/main` → conda/pip sync and prints
the deployed commit. It never touches `server\.env` or the SQLite DB.

1. [ ] Run it once now; it ends with `Deployed:` and a commit line.

> ⚠️ **Do not run `deploy.bat` during an unattended collection window** (e.g. the
> July eval freeze). Pulling a half-finished commit mid-window corrupts the
> frozen set. Freeze the code for the duration and deploy only between windows.
> Never wire `git pull` into `run_collectors.bat`.

---

## Step 11 — Liveness alarm / dead-man's switch (~10 min)

The collector can "succeed" while ingesting zero rows (a rejected key, an empty
GFS cycle), and a crashed scheduler or a slept laptop leaves no error at all.
`run_liveness.bat` checks the age of the newest row per source and pings a free
external monitor **only when everything is fresh** — so a stall, a crash, or a
dead box stops the pings and the monitor emails you. That last case (whole box
down) is the one a local emailer can never catch.

1. [ ] At <https://healthchecks.io> (free), create a check. Set **period ≈ 1h**
       and **grace ≈ 1h**, and copy its ping URL (`https://hc-ping.com/<uuid>`).
2. [ ] Store the URL persistently — it survives reboots *and* git resets, and
       never enters the repo. In a normal Command Prompt:
       ```bat
       setx HEALTHCHECK_URL "https://hc-ping.com/<your-uuid>"
       ```
       Then **open a new prompt** so the variable is in the environment.
3. [ ] Test it by hand:
       ```bat
       C:\temp\aeris\aeris\deploy\windows-collector\run_liveness.bat
       notepad "C:\aeris-data\logs\liveness.log"
       ```
       Expect four `ok` lines, `exit_code=0`, `healthcheck pinged` — and the
       Healthchecks dashboard turns green.
4. [ ] Schedule it every 30 minutes (Administrator Command Prompt):
       ```bat
       schtasks /create /tn "AERIS Liveness" /tr "C:\temp\aeris\aeris\deploy\windows-collector\run_liveness.bat" /sc minute /mo 30 /rl HIGHEST /f
       ```
5. [ ] Force one scheduled run and re-check the log + dashboard:
       ```bat
       schtasks /run /tn "AERIS Liveness"
       ```
6. [ ] **Prove the alarm fires.** Force a stale verdict — the ping is skipped,
       so the check goes red after the grace period, then self-heals on the next
       healthy run (read-only; changes no data):
       ```bat
       cd C:\temp\aeris\aeris\server
       "%USERPROFILE%\miniforge3\envs\aeris\python.exe" -m app.monitoring.liveness --max-age-minutes 1
       ```
       `exit_code=1` + `LIVENESS ALARM` = working.

Per-source freshness budgets: openaq/openweather 3 h, noaa_gfs 12 h, sentinel5p
72 h (it is orbital — ~1 overpass/day, with cloud-gapped days). A `STALE`
sentinel5p inside a couple of days is usually a real collection gap, not a false
alarm.

---

## Monitoring from the road

The Healthchecks email (Step 11) is your primary signal — green means all four
sources are landing fresh data; a red/alert means investigate. If you also want
to eyeball the logs, point the Acer's OneDrive at `C:\aeris-data\logs\` (or copy
the folder there) so `collector.log` and `liveness.log` sync to your phone. A
healthy `collector.log` gains one `===== RUN =====` block per hour.

---

## Eval freeze checklist — 2026 (Jul 1–13)

The unattended collection window is **Jul 1–12**; the eval set freezes **Jul 13**
from the Jun 1 → Jul 12 data. Run this on the last day you have hands on the box.

**Before Jul 1 — lock it down**

1. [ ] Run `deploy.bat` once; confirm the deployed commit is the one you intend
       to freeze on. This is the last deploy until after the freeze.
2. [ ] `cd C:\temp\aeris\aeris\server` then `python -m app.collectors.run_all`
       **twice** — four `ok` lines, and still all `ok` on the rerun (dedup works).
3. [ ] Healthchecks is green; the force-stale test (Step 11.6) turns it red, then
       it heals on the next healthy run.
4. [ ] Windows Update paused through Jul 13 (re-click to extend the 7-day cap).
5. [ ] AC power + Ethernet; sleep/hibernate off (Step 6); signed in; lid-close
       does nothing.
6. [ ] `dir C:\aeris-data` shows enough free disk for ~2 weeks of rows.

**Jul 1–12 — hands off**

7. [ ] **Do not run `deploy.bat`, `git pull`, or edit anything.** The code is frozen.
8. [ ] If Healthchecks alerts, fix the smallest thing (a key, power, network).
       Avoid redeploying mid-window unless collection has gone fully dark.

**Jul 13 — freeze handoff**

9.  [ ] Copy `C:\aeris-data\aeris.db` to the Mac.
10. [ ] On the Mac, build the fixture:
        `python -m app.eval.freeze --start 2026-06-01 --end 2026-07-12 --top 50 --out fixtures/eval50.json`.
11. [ ] Before trusting the set, scan `liveness.log` / the Healthchecks history
        for any gap inside the window.

---

## Troubleshooting

**`conda env create` fails on `pygrib`/`eccodes`.**
The other three collectors don't need it. Edit `environment.yml`, delete the
`pygrib` and `eccodes` lines, and re-run Step 2. NOAA GFS will report `failed`
each run — that's fine: NOMADS keeps ~10 days of GFS cycles, so you can
backfill the gap with `python -m app.collectors.backfill` when you return
(do it the first day back, before the early days roll off the NOMADS window).

**`git clone`/`deploy.bat` keeps asking for a password or fails auth.**
The password must be a fine-grained **PAT** (Contents: Read-only on `aeris`),
not your account password. Re-run `git config --global credential.helper store`,
then clone again and paste the PAT once.

**`deploy.bat` says "not a git checkout".**
The folder was copied (USB/zip), not cloned. Run the one-time in-place migration
in Step 0's callout to convert it.

**Liveness task never pings / Healthchecks stays red.**
`HEALTHCHECK_URL` wasn't in the scheduled session — confirm `setx` ran, then let
the next run pick it up (or recreate the task). Confirm `curl https://hc-ping.com/<uuid>`
works from a prompt. A genuinely stale source also keeps it red until data flows.

**Task didn't run / log not updating.**
Open Task Scheduler (`taskschd.msc`), find the task, check **History**. Most
common cause: the Acer slept (redo Step 6) or got signed out (stay logged in, or
use the Step 8 hardening variant).

**A collector reports `failed` with an auth error.**
A key in `server\.env` is wrong or missing. The other collectors keep working.

**Moving the data to your Mac later.**
`C:\aeris-data\aeris.db` is a single self-contained file. Copy it over, point
your Mac's `.env` `DATABASE_URL` at it, and run `python -m app.detection.run`.

---

## When you get back

1. Check the Healthchecks history and `liveness.log` / `collector.log` for gaps.
2. If GFS was disabled, run `python -m app.collectors.backfill` **first** —
   the NOMADS window is only ~10 days.
3. Copy `aeris.db` to your Mac (or keep running detection on the Acer).
4. Run detection on the accumulated data.
