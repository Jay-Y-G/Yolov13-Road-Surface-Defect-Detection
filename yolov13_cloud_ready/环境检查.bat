@echo off
call cd/d "%~dp0"
call conda activate yolov13
call python environmnet_check.py
pause