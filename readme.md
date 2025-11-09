python -m venv .venv

# mac/linux

source .venv/bin/activate
# windows

.venv\Scripts\activate

pip install -r requirements.txt

# Run script
python auto_chart.py "path/to/song.wav" --out beatmap.json --subdiv 4 --downbeat-every 4 --demucs

# Run as compose
docker-compose run --rm auto-chart /input/1_believer.mp3 --out /output/believer_beatmap.json --subdiv 4 --downbeat-every 4 --demucs