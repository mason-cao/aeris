@echo off
REM AERIS liveness alarm - invoked every 30-60 min by Windows Task Scheduler.
REM Exits 0 when every source's newest row is within its freshness budget, and
REM pings the dead-man's-switch URL. On a stall (exit 1) or a dead box the ping
REM stops, so the external monitor alerts you - the one failure mode a local
REM emailer cannot catch.
REM
REM Set the ping URL once, persistently (kept OUT of the repo so it survives a
REM git reset and never leaks):
REM   setx HEALTHCHECK_URL "https://hc-ping.com/<your-uuid>"
REM Edit the three paths below only if they differ from the SETUP.md defaults.

set "PYTHON=%USERPROFILE%\miniforge3\envs\aeris\python.exe"
set "REPO=C:\temp\aeris\aeris"
set "LOGDIR=C:\aeris-data\logs"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set "LOG=%LOGDIR%\liveness.log"

echo. >> "%LOG%"
echo ===== LIVENESS %DATE% %TIME% ===== >> "%LOG%"

if not exist "%PYTHON%" (
    echo ERROR: python not found at %PYTHON% >> "%LOG%"
    exit /b 1
)
if not exist "%REPO%\server" (
    echo ERROR: repo not found at %REPO%\server >> "%LOG%"
    exit /b 1
)

cd /d "%REPO%\server"
"%PYTHON%" -m app.monitoring.liveness >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
echo exit_code=%RC% >> "%LOG%"

if "%RC%"=="0" (
    if defined HEALTHCHECK_URL (
        curl -fsS -m 10 --retry 3 "%HEALTHCHECK_URL%" >> "%LOG%" 2>&1
        echo healthcheck pinged >> "%LOG%"
    ) else (
        echo HEALTHCHECK_URL not set - skipping ping >> "%LOG%"
    )
) else (
    echo LIVENESS ALARM: stale source detected - ping skipped >> "%LOG%"
)
exit /b %RC%
