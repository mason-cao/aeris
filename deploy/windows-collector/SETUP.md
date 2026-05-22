# AERIS Windows Collector — Setup

Turns the Acer (i7-4710HQ / 16 GB / Win 10 Education) into an unattended data
collector that runs all four AERIS collectors hourly into a local SQLite
database. Built to survive a one-week travel gap with no one at the keyboard.

**Total time: ~75 minutes.** Do the steps in order. Boxes to tick are marked `[ ]`.

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
- **A scheduled task** — runs `run_collectors.bat` every hour.

Database is **SQLite** at `C:\aeris-data\aeris.db` — one portable file, no
service to crash. Copy that file to your Mac when you're back to run detection.

---

## Step 0 — Get the code onto the Acer (~10 min)

The fastest, most reliable path is a **USB copy** — it carries `server\.env`
(your API keys), which is gitignored and would not come via `git clone`.

1. [ ] On your Mac, copy the whole `aeris` project folder to a USB stick.
       You can skip `server/venv/` if it exists (a Mac venv is useless here).
2. [ ] On the Acer, paste it to **`C:\aeris`** so the layout is
       `C:\aeris\server`, `C:\aeris\deploy`, etc.
3. [ ] Confirm `C:\aeris\server\.env` exists and contains your real API keys
       (`OPENAQ_API_KEY`, `OPENWEATHER_API_KEY`, `NASA_EARTHDATA_TOKEN`,
       `CDSE_USERNAME`, `CDSE_PASSWORD`).

*Alternative:* `git clone https://github.com/mason-cao/aeris.git C:\aeris` — but
you must still copy `server\.env` separately, and you'll need GitHub auth.

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
conda env create -f C:\aeris\deploy\windows-collector\environment.yml
conda activate aeris
pip install -r C:\aeris\deploy\windows-collector\requirements-collector.txt
```

1. [ ] `conda env create` finishes without error (this installs `pygrib` +
       `eccodes` — the part that would fail under plain pip).
2. [ ] `pip install` finishes without error.

If `conda env create` fails on `pygrib`, see **Troubleshooting** — the other
three collectors still work and you can move on.

---

## Step 3 — Point the config at SQLite (~5 min)

Edit **`C:\aeris\server\.env`** in Notepad. Change the `DATABASE_URL` line and
the `AERIS_ENV` line to exactly:

```
DATABASE_URL=sqlite+aiosqlite:///C:/aeris-data/aeris.db
AERIS_ENV=production
```

- `sqlite+aiosqlite` — async SQLite driver, already installed.
- Forward slashes are correct even on Windows.
- `AERIS_ENV=production` silences SQLAlchemy's per-query echo so the log stays small.

1. [ ] `.env` saved with both lines changed.

---

## Step 4 — Create the database (~3 min)

In the Miniforge Prompt (env still active):

```bat
mkdir C:\aeris-data
cd C:\aeris
python deploy\windows-collector\init_db.py
```

1. [ ] Output ends with `Tables ready in sqlite+aiosqlite:///C:/aeris-data/aeris.db`.

---

## Step 5 — Test a real collection run (~5 min)

```bat
cd C:\aeris\server
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
2. [ ] **Leave the Acer plugged into AC power** for the whole week.
3. [ ] Prefer an **Ethernet cable** over Wi-Fi for a stable week-long connection.
       On Wi-Fi, confirm the network is set to "Connect automatically".

---

## Step 7 — Pause Windows Update (~3 min)

A forced update reboot mid-week would interrupt collection.

1. [ ] **Settings → Update & Security → Windows Update → Pause updates for 7 days.**
       Click it again if it lets you extend further.

The scheduled task survives a reboot anyway (Step 8), so a reboot costs at most
one hour of data — but pausing avoids it entirely.

---

## Step 8 — Schedule the hourly task (~5 min)

In the **Administrator Command Prompt**:

```bat
schtasks /create /tn "AERIS Collector" /tr "C:\aeris\deploy\windows-collector\run_collectors.bat" /sc hourly /rl HIGHEST /f
```

1. [ ] Output says `SUCCESS: The scheduled task "AERIS Collector" ...`.

This runs every hour **while you are logged in**. For the week away, that's
fine — just **leave the Acer logged in** (don't sign out; the lid can be closed).

*Optional hardening* — to run even through a reboot to the login screen, recreate
the task with your account + password (only works if your account has a password):

```bat
schtasks /create /tn "AERIS Collector" /tr "C:\aeris\deploy\windows-collector\run_collectors.bat" /sc hourly /rl HIGHEST /ru "%USERNAME%" /rp * /f
```

It will prompt for your Windows password.

---

## Step 9 — Verify the task works (~5 min)

```bat
schtasks /run /tn "AERIS Collector"
```

Wait ~30 seconds, then open the log:

```bat
notepad "%USERPROFILE%\OneDrive\aeris-logs\collector.log"
```

1. [ ] The log shows a `===== RUN ... =====` block, four collector lines, and
       `exit_code=0` (or `exit_code=1` only if GFS is the lone failure).

You're done. The task will fire at the top of every hour.

---

## Step 10 — Monitoring from vacation (optional, ~5 min)

The log is written to `%USERPROFILE%\OneDrive\aeris-logs\collector.log`. If the
Acer is signed into OneDrive, that file syncs to the cloud — **open the OneDrive
app on your phone to read it from anywhere**. A healthy log gains one
`===== RUN =====` block per hour.

If OneDrive isn't set up and you don't have 5 minutes, skip it — the data is
still safe locally in `C:\aeris-data\aeris.db`.

---

## Troubleshooting

**`conda env create` fails on `pygrib`/`eccodes`.**
The other three collectors don't need it. Edit `environment.yml`, delete the
`pygrib` and `eccodes` lines, and re-run Step 2. NOAA GFS will report `failed`
each run — that's fine: NOMADS keeps ~10 days of GFS cycles, so you can
backfill the gap with `python -m app.collectors.backfill` when you return
(do it the first day back, before the early days roll off the NOMADS window).

**Task didn't run / log not updating.**
Open Task Scheduler (`taskschd.msc`), find "AERIS Collector", check
**History**. Most common cause: the Acer slept (redo Step 6) or got signed out
(stay logged in, or use the Step 8 hardening variant).

**A collector reports `failed` with an auth error.**
A key in `server\.env` is wrong or missing. The other collectors keep working.

**Moving the data to your Mac later.**
`C:\aeris-data\aeris.db` is a single self-contained file. Copy it over, point
your Mac's `.env` `DATABASE_URL` at it, and run `python -m app.detection.run`.

---

## When you get back (May 29)

1. Check `collector.log` for gaps.
2. If GFS was disabled, run `python -m app.collectors.backfill` **first** —
   the NOMADS window is only ~10 days.
3. Copy `aeris.db` to your Mac (or keep running detection on the Acer).
4. Run detection on the accumulated week of data.
