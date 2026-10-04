@echo off
title Windrecorder
mode con cols=70 lines=10
color 75
echo.
echo   Initializing Windrecorder, please stand by...
echo.
echo   Please stay in this window until it disappears
echo.

cd /d %~dp0
if exist "hide_CLI_by_python.txt" (
    goto begin
) else (
    goto hide
)

:hide
@REM hide CLI immediately
if "%1"=="h" goto begin
@REM After Windows 11 26200+ update, mshta's inline vbscript: protocol no longer
@REM spawns child processes. Use wscript with a temp .vbs instead (mode 0 = hidden window).
> "%TEMP%\wr_hide.vbs" echo Set s=CreateObject("WScript.Shell"):s.Run "%~f0 h",0,FALSE
wscript "%TEMP%\wr_hide.vbs"
exit /b

:begin
cd /d %~dp0
call conda activate py311

set SSL_CERT_FILE=%CONDA_PREFIX%\Lib\site-packages\certifi\cacert.pem

python "%~dp0\main.py"