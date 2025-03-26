from pathlib import Path
from typing import Optional
import numpy as np
from fastapi import HTTPException

from ....core.series_processor import SeriesProcessor
from ....core.processing_params import ProcessingParams

class ProcessorService:
    def __init__(self):
        self._processor: Optional[SeriesProcessor] = None

    @property
    def processor(self) -> SeriesProcessor:
        if not self._processor:
            raise HTTPException(status_code=400, detail="No data loaded")
        return self._processor

    def load_from_path(self, path: str) -> tuple[str, int]:
        """Load XRD image series from a path."""
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                raise ValueError(f"Path does not exist: {path}")
            
            self._processor = SeriesProcessor(str(path_obj))
            return f"Loaded {self._processor.img_num} images from {path}", self._processor.img_num
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    def process_images(self, params: ProcessingParams) -> str:
        """Process images with given parameters."""
        try:
            self.processor.load_params(params)
            self.processor.avg_clean_img()
            return "Processing completed"
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    def get_plot_data(self, plot_type: str) -> np.ndarray:
        """Get plot data for visualization."""
        try:
            data = None
            if plot_type == "average":
                data = self.processor.img_avg
            elif plot_type == "clean":
                data = self.processor.img_clean_avg
            elif plot_type == "difference":
                data = self.processor.img_diff_avg
            elif plot_type == "mask":
                data = self.processor.ring_mask.astype(np.int32)
            elif plot_type == "donut":
                data = self.processor.sub_donut_avg
            elif plot_type == "streak":
                data = self.processor.sub_streak_avg
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")
            
            return data
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) 