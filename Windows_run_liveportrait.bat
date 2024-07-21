@echo off
call LivePortrait_env\Scripts\activate
python app.py

timeout /t 5
start http://127.0.0.1:8890/