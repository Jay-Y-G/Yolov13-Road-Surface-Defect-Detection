@echo off
call conda activate labelimg
call cd/d "%~dp0\data"
call labelimg JPEGImages classes.txt
pause