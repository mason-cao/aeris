@echo off
REM Pull the latest AERIS code from GitHub onto this collection host.
REM RUN FROM THE MINIFORGE PROMPT (needs git + conda on PATH).
REM Safe to re-run. Does NOT touch server\.env or C:\aeris-data\aeris.db
REM (gitignored / outside the tree), so your API keys and data survive every pull.
REM Never add "git clean" here - it would delete those untracked files.
setlocal
cd /d "%~dp0..\.." || (echo ERROR: cannot locate repo root & exit /b 1)
echo Repo root: %CD%

git rev-parse --is-inside-work-tree >nul 2>&1 || (echo ERROR: not a git checkout - do the one-time migration first ^(see SETUP.md^) & exit /b 1)

echo Fetching latest...
git fetch origin || (echo ERROR: git fetch failed - check network / GitHub auth & exit /b 1)
git reset --hard origin/main || (echo ERROR: git reset failed & exit /b 1)

echo Syncing dependencies ^(no-op when unchanged^)...
call conda env update -n aeris -f deploy\windows-collector\environment.yml
call conda run -n aeris pip install -r deploy\windows-collector\requirements-collector.txt

echo.
echo Deployed:
git log --oneline -1
endlocal
