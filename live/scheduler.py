import sys
import os
sys.path.insert(0, '/app')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import schedule
import time

# Import directly instead of from live.paper_trader
from paper_trader import run

# 9:30 AM EST = 7:00 PM IST
schedule.every().monday.at("19:00").do(run)
schedule.every().tuesday.at("19:00").do(run)
schedule.every().wednesday.at("19:00").do(run)
schedule.every().thursday.at("19:00").do(run)
schedule.every().friday.at("19:00").do(run)

print("🚀 Scheduler running — agent will trade at 7:00 PM IST on weekdays")

while True:
    schedule.run_pending()
    time.sleep(60)
