import sys
sys.path.insert(0, '/Users/anshvardhansingh/Desktop/trading_agent')

import schedule
import time
from live.paper_trader import run

# Run every weekday at market open (adjust for your timezone)
# 9:30 AM EST = 7:00 PM IST
schedule.every().monday.at("19:00").do(run)
schedule.every().tuesday.at("19:00").do(run)
schedule.every().wednesday.at("19:00").do(run)
schedule.every().thursday.at("19:00").do(run)
schedule.every().friday.at("19:00").do(run)

print("🚀 Scheduler running — agent will trade at 7:00 PM IST on weekdays")
print("   Press Ctrl+C to stop\n")

while True:
    schedule.run_pending()
    time.sleep(60)
