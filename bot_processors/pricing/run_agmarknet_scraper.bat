@echo off
cd /d "C:\Users\Praneeth\Desktop\Agent\pipecat"
"C:\Users\Praneeth\Desktop\Agent\pipecat\venv\Scripts\python.exe" -m bot_processors.pricing.agmarknet_scraper >> "C:\Users\Praneeth\Desktop\Agent\pipecat\bot_processors\pricing\scraper_log.txt" 2>&1
