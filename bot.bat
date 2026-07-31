@echo off
REM Always runs Bot.py with the project's venv Python, regardless of
REM whether the venv is activated or what "py"/"python" resolve to in
REM this terminal. Avoids the rank_bm25/rapidfuzz "module not found"
REM errors caused by py.exe falling back to the system Python install.
"%~dp0venv\Scripts\python.exe" "%~dp0Bot.py" %*
