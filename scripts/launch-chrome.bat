@echo off
REM Launch Chrome on Windows with a persistent profile + CDP debug port so
REM the agent's BROWSER_GO / BROWSER_CLICK / BROWSER_FILL fast-path can
REM drive the active tab without going through the VLM.
REM
REM Usage:
REM     scripts\launch-chrome.bat
REM     scripts\launch-chrome.bat https://youtube.com
REM
REM Override the profile dir / port via environment variables:
REM     set PROFILE_DIR=C:\Users\you\chrome-profile
REM     set CDP_PORT=29229
REM
REM Set CDP_PORT=0 to skip the debug flag entirely.

setlocal

if "%PROFILE_DIR%"=="" set PROFILE_DIR=%LOCALAPPDATA%\desktop-vlm-agent\chrome-profile
if "%CDP_PORT%"=="" set CDP_PORT=29229

set START_URL=%1
if "%START_URL%"=="" set START_URL=about:blank

if not exist "%PROFILE_DIR%" mkdir "%PROFILE_DIR%"

REM Try the standard install paths; the user can override with CHROME=...
if "%CHROME%"=="" (
    if exist "%PROGRAMFILES%\Google\Chrome\Application\chrome.exe" (
        set CHROME=%PROGRAMFILES%\Google\Chrome\Application\chrome.exe
    ) else if exist "%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe" (
        set CHROME=%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe
    ) else if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" (
        set CHROME=%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe
    ) else (
        echo error: chrome.exe not found in standard locations.
        echo set CHROME=path\to\chrome.exe to override.
        exit /b 1
    )
)

REM --remote-allow-origins is REQUIRED on Chrome 111+ or Chrome rejects
REM the websocket handshake with 403 Forbidden. http://localhost:<port>
REM is the narrowest allow-list that still lets BrowserBridge connect.
set DEBUG_FLAG=
if not "%CDP_PORT%"=="0" set DEBUG_FLAG=--remote-debugging-port=%CDP_PORT% --remote-allow-origins=http://localhost:%CDP_PORT%

echo [launch-chrome] profile=%PROFILE_DIR% url=%START_URL% cdp-port=%CDP_PORT%
"%CHROME%" --user-data-dir="%PROFILE_DIR%" --no-first-run --no-default-browser-check %DEBUG_FLAG% "%START_URL%"
