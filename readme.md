python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows
# .venv\Scripts\activate

pip install -r requirements.txt
# optional (for high-quality stems):
pip install demucs torch

python auto_chart.py "path/to/song.wav" --out beatmap.json --subdiv 4 --downbeat-every 4 --demucs
