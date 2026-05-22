@echo off
REM AERIS data collector - invoked hourly by Windows Task Scheduler.
REM Edit the three paths below only if you did not use the SETUP.md defaults.

set "PYTHON=%USERPROFILE%\miniforge3\envs\aeris\python.exe"
set "REPO=C:\temp\aeris\aeris"
set "LOGDIR=%USERPROFILE%\OneDrive\aeris-logs"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set "LOG=%LOGDIR%\collector.log"

echo. >> "%LOG%"
echo ===== RUN %DATE% %TIME% ===== >> "%LOG%"

if not exist "%PYTHON%" (
    echo ERROR: python not found at %PYTHON% >> "%LOG%"
    exit /b 1
)
if not exist "%REPO%\server" (
    echo ERROR: repo not found at %REPO%\server >> "%LOG%"
    exit /b 1
)

cd /d "%REPO%\server"
"%PYTHON%" -m app.collectors.run_all >> "%LOG%" 2>&1
echo exit_code=%ERRORLEVEL% >> "%LOG%"
