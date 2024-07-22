@echo off
call LivePortrait_env\Scripts\activate

:: Start the Python server
start /B python app.py

:: Wait for the server to start (increased to 10 seconds for slower systems)
echo Waiting for server to start...
timeout /t 10

:: Open the browser
start http://127.0.0.1:8890/

:: Keep the batch file running and show server output
echo Server is running. Press Ctrl+C to stop.
:loop
timeout /t 1 >nul
goto loop
