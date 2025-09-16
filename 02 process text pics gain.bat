@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Automontage — Merged Pipeline

echo =======================================================
echo == Starting merged pipeline
echo =======================================================

REM Go to project folder
pushd "D:\project\archiver\prog\automontage" || (
  echo [ERROR] Could not change directory to D:\project\archiver\prog\automontage
  pause & exit /b 1
)

REM Capture the system Python *before* activating any venv
set "SYS_PY="
for /f "delims=" %%P in ('where python 2^>nul') do (
  if not defined SYS_PY set "SYS_PY=%%~fP"
)
if not defined SYS_PY (
  echo [ERROR] Could not find system Python on PATH before venv activation.
  echo         Make sure "python" works in a fresh CMD in this folder.
  goto :cleanup_fail
)
echo [INFO] System Python detected: "%SYS_PY%"

REM Activate venv for Part 1
if exist "venv_chain\Scripts\activate.bat" (
  echo [INFO] Activating virtual environment: venv_chain
  call "venv_chain\Scripts\activate.bat"
) else (
  echo [WARN] venv_chain\Scripts\activate.bat not found. Using system Python for Part 1.
)

echo.
echo === Part 1: addVectorsToText (venv or system) ===
python "02 addVectorsToText.py"
if errorlevel 1 goto :fail_addVectorsToText
echo --- Finished addVectorsToText
echo.

echo === Part 2: addPictureVectors (forced system Python) ===
"%SYS_PY%" "03 addPictureVectors.py"
if errorlevel 1 goto :fail_addPictureVectors
echo --- Finished addPictureVectors
echo.

echo === Part 3: volumeInfos (forced system Python) ===
"%SYS_PY%" "04 volumeInfos.py"
if errorlevel 1 goto :fail_volumeInfos
echo --- Finished volumeInfos
echo.

echo =======================================================
echo == All tasks completed successfully
echo =======================================================
popd
pause
exit /b 0

:fail_addVectorsToText
echo [ERROR] addVectorsToText failed with exit code %ERRORLEVEL%.
goto :cleanup_fail

:fail_addPictureVectors
echo [ERROR] addPictureVectors failed with exit code %ERRORLEVEL%.
goto :cleanup_fail

:fail_volumeInfos
echo [ERROR] volumeInfos failed with exit code %ERRORLEVEL%.
goto :cleanup_fail

:cleanup_fail
popd
pause
exit /b %ERRORLEVEL%
