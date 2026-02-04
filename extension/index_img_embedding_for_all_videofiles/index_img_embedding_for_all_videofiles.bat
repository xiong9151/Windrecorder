@echo off
echo Loading extension, please stand by.
echo.

cd /d %~dp0
call conda activate py311
chcp 65001
cls

python "%~dp0\index_img_embedding_for_all_videofiles.py"
pause