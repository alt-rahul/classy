# Paused (2026-11-20 22:05 UTC)

State at pause: store and submissions/submission.csv at 133,862 (validated, invalid 0); best Kaggle score 133,862 (ref 58636021).

Stopped: the Kaggle compute agent, both round-5 research agents (research/round5/constructor, research/round5/structured), the submit daemon (src/submit_daemon.sh), the hourly public refresh (src/refresh_loop.sh), the Kaggle fleet controller (kaggle/fleet.py loop) and all local compute jobs.

Still on Kaggle: 5 private notebook sessions launched before the pause (slots elf-c01..c05 in store/kaggle_state.json; 4 x lever-B group rebuild with 3.5 h budgets, 1 x public LNS with 2.5 h). They end on their own by about 01:30 UTC; nothing new is launched while the controller is stopped. Their outputs stay on Kaggle and are harvested by the controller on resume.

Resume:
  cd /home/user/classy/elf
  nohup bash src/submit_daemon.sh > /dev/null 2>&1 &
  nohup bash src/refresh_loop.sh > /dev/null 2>&1 &
  nohup python3 kaggle/fleet.py loop --every 300 >> store/logs/kaggle_fleet.out 2>&1 &
Research notes to pick up from: research/round5/constructor/NOTES.md, research/round5/structured/NOTES.md.
