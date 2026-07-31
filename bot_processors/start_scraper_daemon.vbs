Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\prane\OneDrive\Desktop\pipecat"
WshShell.Run """C:\Users\prane\OneDrive\Desktop\pipecat\venv\Scripts\python.exe"" -m bot_processors.scraper_daemon", 0, False
