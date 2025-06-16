from pathlib import Path
from xrd_decosmic.core.series_processor import SeriesProcessor, SeriesConfig

RAW_FOLDER = "data/raw"
SAVE_FOLDER = "data/processed"
SAVE_NAME_PREFIX = "popc"

raw_files = sorted(list(Path(RAW_FOLDER).expanduser().glob("*.tif")))

series_config = SeriesConfig(
    th_donut=15,
    th_mask=0.05,
    th_streak=3,
    win_streak=3,
    exp_donut=9,
    exp_streak=3
)

processor = SeriesProcessor(
    str(raw_files[0]),
    series_config,
)

series_result = processor.process_series()

series_result.save(SAVE_FOLDER, SAVE_NAME_PREFIX)