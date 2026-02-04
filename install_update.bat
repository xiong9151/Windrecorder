@echo off
title Windrecorder - installing dependence
mode con cols=150 lines=50

cd /d %~dp0

echo -activating conda environment py311
call conda activate py311

echo -upgrading pip and ensuring Python 3.11
python --version
python -m pip install --upgrade pip setuptools

echo -installing poetry with清华源
python -m pip install poetry -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple poetry

echo -configuring poetry to use project-local virtual environment
poetry config virtualenvs.in-project true

echo -clearing poetry cache to avoid conflicts
poetry cache clear pypi --all

echo -installing dependencies with --no-root flag to avoid package conflicts
poetry install --no-root --verbose

if %errorlevel% neq 0 (
    echo Installation failed, attempting to install with --sync flag...
    poetry install --no-root --sync
)

if %errorlevel% neq 0 (
    echo Installation still failed, attempting to install with --only=main flag...
    poetry install --only=main
)

echo -verifying critical packages installation
python -c "import cv2; print('OpenCV version: ' + cv2.__version__)"
if %errorlevel% neq 0 (
    echo Installing OpenCV separately...
    pip install opencv-python==4.8.1.78 -i https://pypi.tuna.tsinghua.edu.cn/simple
)

echo -activating poetry environment
for /F "usebackq tokens=*" %%A in (`poetry env info --path`) do call "%%A\Scripts\activate.bat"

color 0e
title Windrecorder - Quick Setup
echo Running onboard setting...
python "%~dp0\onboard_setting.py"

pause