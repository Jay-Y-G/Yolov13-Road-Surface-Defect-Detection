@echo off
call cd/d "%~dp0"
call conda  activate yolov13
call python "picture_detect.py"
pause