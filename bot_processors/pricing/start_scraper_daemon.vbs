Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\Praneeth\Desktop\Agent\pipecat"
' Routed through cmd /c so stdout/stderr (loguru's default sink) land in a
' real log file instead of vanishing into an unobservable hidden window --
' confirmed live (2026-08-10): launching python.exe directly here had no
' way to tell, short of opening the hidden window, whether the daemon was
' actually running or had silently died on a startup error.
WshShell.Run "cmd /c """"C:\Users\Praneeth\Desktop\Agent\pipecat\venv\Scripts\python.exe"" -m bot_processors.pricing.scraper_daemon >> ""C:\Users\Praneeth\Desktop\Agent\pipecat\bot_processors\pricing\scraper_daemon_log.txt"" 2>&1""", 0, False
