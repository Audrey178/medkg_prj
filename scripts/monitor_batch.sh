#!/bin/bash
# Persistent batch monitor for ChronoMedKG pipeline
# Runs every 10 minutes via launchd/cron
# Logs to /tmp/primekg_monitor.log

LOG="/tmp/primekg_batch_17k.log"
MONITOR_LOG="/tmp/primekg_monitor.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "=== Monitor check: $TIMESTAMP ===" >> "$MONITOR_LOG"

# Check if process is alive
PIDS=$(pgrep -f "orchestrator.py.*batch_next")
if [ -z "$PIDS" ]; then
    echo "⚠️  ALERT: No orchestrator process found! Batch may have crashed or completed." >> "$MONITOR_LOG"
    # Send macOS notification
    osascript -e 'display notification "Batch process not found! Check logs." with title "ChronoMedKG Monitor" sound name "Basso"' 2>/dev/null
else
    echo "✅ Process alive: PIDs=$PIDS" >> "$MONITOR_LOG"
fi

# Check for failures
FAILURES=$(grep -c "PIPELINE FAILED" "$LOG" 2>/dev/null)
FAILURES=${FAILURES:-0}
if [ "$FAILURES" -gt 0 ]; then
    echo "⚠️  ALERT: $FAILURES disease pipeline(s) FAILED!" >> "$MONITOR_LOG"
    grep "PIPELINE FAILED" "$LOG" | tail -5 >> "$MONITOR_LOG"
    osascript -e "display notification \"$FAILURES disease(s) FAILED! Check logs.\" with title \"ChronoMedKG Monitor\" sound name \"Basso\"" 2>/dev/null
fi

# Progress summary
SUCCESSES=$(grep -c "PIPELINE SUCCESS" "$LOG" 2>/dev/null)
SUCCESSES=${SUCCESSES:-0}
LATEST_DOC=$(grep "Document [0-9]" "$LOG" 2>/dev/null | tail -1)
echo "Completed: $SUCCESSES | Failed: $FAILURES | Latest: $LATEST_DOC" >> "$MONITOR_LOG"
echo "" >> "$MONITOR_LOG"
