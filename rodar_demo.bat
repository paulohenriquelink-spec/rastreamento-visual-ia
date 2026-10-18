@echo off
rem Roda a demo com o Python do ambiente do projeto (.venv), que tem OpenCV e YOLO.
cd /d "%~dp0"
".venv\Scripts\python.exe" demo.py %*
pause
