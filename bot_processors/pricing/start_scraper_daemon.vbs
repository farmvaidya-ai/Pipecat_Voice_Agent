Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\Praneeth\Desktop\Agent\pipecat"
WshShell.Run """C:\Users\Praneeth\Desktop\Agent\pipecat\venv\Scripts\python.exe"" -m bot_processors.pricing.scraper_daemon", 0, False
