## Create Environment
Create a conda env by run: 
```
conda create -n surg_gui python=3.11 -y
conda activate surg_gui
pip install fastapi uvicorn[standard] numpy pandas python-multipart opencv-python pyarrow lerobot==0.3.3
```

The core libraries are fastapi and uvicorn. Feel free to add other libraries if I miss anything.

## A. Sync All Modalities
1. Fill the .json file like
```json
{
  "timezone": "America/Vancouver",
  "left_ts": "path/to/left_ts.txt",
  "right_ts": "path/to/right_ts.txt",
  "side_ts": "path/to/side_ts.txt",
  "gaze_log": "path/to/gazelog.txt",
  "dvapi_csv": "path/to/dvapi.csv"
}
```
2. Then run 
```
python ./sync_all.py --config ./data/example/mm_timestamp.json --out_csv ./data/example/sync_table_all.csv
```
3. The output .csv file is the synchronized timestamps of all modalities.

## B. Trim Left Video Based on Sync Data
This stage is to exclude useless frames for easier clipping later. All frames of the generated left proxy video can be retrieved in `sync_table_all.csv` after this stage.

You need: (1) sync file `sync_table_all.csv` from Stage A; (2) raw left video (you can download [here](https://drive.google.com/file/d/12JozZDyQ9tZ60jvYP73CLpt3HmZ9byMm/view?usp=sharing) and put it in ./data/example). 

Run:
```
python ./trim_left_video.py --sync_csv [SYNC FILE PATH] --video [RAW LEFT VIDEO PATH] --out_video [PROXY VIDEO PATH] --out_map [OUTPUT MAP FILE] --frames_dir [EXTRACT FRAMES DIR] --idx_col left_idx --ts_col t_ref_s
```
For example:
```
python ./trim_left_video.py --sync_csv ./data/example/sync_table_all.csv --video ./data/example/left_raw_video.mp4 --out_video ./data/example/proxy_left.mp4 --out_map ./data/example/proxy_left_index_map.csv --frames_dir ./data/example/proxy_left_frames --idx_col left_idx --ts_col t_ref_s
```

If you do not need the extracted frames, remove --frames_dir [EXTRACT FRAMES DIR].

## C. Run GUI
1. Run `ssh -L 8000:127.0.0.1:8000 user@server`, where change `user@server` as yours.
2. Run `uvicorn app:app --host 127.0.0.1 --port 8000` on your server.
3. Then you should be able to open `http://localhost:8000/docs` in the local browser.

Note that all path need to be server path.

## D. Clip Episodes
The usage of this GUI is very straightforward.

## E. Export Data With Label

## F. Convert to LeRobot Format