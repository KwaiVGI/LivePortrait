@echo off
call LivePortrait_env\Scripts\activate

:: Start the Python server in a new window
start "Python Server" cmd /k python app.py

:: Wait for the server to start (adjust the time if needed)
timeout /t 5

:: Open the browser
start http://127.0.0.1:8890/

:: Keep the batch file running
echo Server is running. Press Ctrl+C to stop.
:loop
timeout /t 10 >nul
goto loop
