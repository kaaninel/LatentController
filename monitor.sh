#!/bin/bash
# Simple training monitor — run in any terminal:
#   cd ~/Workspace/LatentController && bash monitor.sh
cd "$(dirname "$0")"
source .venv/bin/activate

while true; do
  clear
  echo "═══ ANT Training Monitor ═══"
  echo ""
  python3 -c "
import torch, time, os
path = 'checkpoints/train/checkpoint_latest.pt'
if not os.path.exists(path):
    print('  No checkpoint found yet.')
else:
    mod = time.time() - os.path.getmtime(path)
    c = torch.load(path, map_location='cpu', weights_only=False)
    step = c.get('step', '?')
    phase = c.get('phase', '?')
    total = 10000
    pct = (step/total)*100 if isinstance(step,int) else 0
    print(f'  Step:  {step} / {total}  ({pct:.1f}%)')
    print(f'  Phase: {phase}')
    print(f'  Saved: {mod:.0f}s ago')
    bar = int(pct/2)
    print(f'  [{\"█\"*bar}{\"░\"*(50-bar)}]')
print('')
print('  (updates every 30s — Ctrl+C to quit)')
"
  sleep 30
done
