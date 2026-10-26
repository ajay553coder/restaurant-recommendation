@echo off
setlocal
echo ===================================================
echo DineWise ML — Personalized Restaurant Discovery
echo ===================================================
echo.

:: Detect Python launcher (prefer py -3.14 if available)
set PYTHON_CMD=python
py -3.14 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3.14
) else (
    py --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=py
    )
)

echo Using Python: %PYTHON_CMD%
%PYTHON_CMD% --version

echo.
echo [Step 1/3] Installing Dependencies...
%PYTHON_CMD% -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Warning or error during pip install. Proceeding with existing environment...
)

echo.
echo [Step 2/3] Checking / Training DWDM ML Models (KNN, K-Means, Decision Tree)...
if not exist "model\metadata.json" (
    %PYTHON_CMD% train_model.py
    if %errorlevel% neq 0 (
        echo Model training failed. Please verify dataset in data/ directory.
        pause
        exit /b %errorlevel%
    )
) else (
    echo Artifacts already exist in model/ directory.
)

echo.
echo [Step 3/3] Launching DineWise Streamlit Web UI...
%PYTHON_CMD% -m streamlit run app.py

pause
