@echo off
cd /d "D:\project\archiver\prog\automontage"
call venv_chain\Scripts\activate
python "02 addVectorsToText.py"
pause
