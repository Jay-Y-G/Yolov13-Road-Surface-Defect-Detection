@echo off
call cd/d "%~dp0"
call conda  activate yolov13
call python "video_detect.py"
pause