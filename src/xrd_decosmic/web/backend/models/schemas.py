from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ProcessingResponse(BaseModel):
    """Response model for processing status."""
    status: str
    message: str

class PathRequest(BaseModel):
    """Request model for loading data from path."""
    path: str

class DirectoryItem(BaseModel):
    """Model for directory browsing item."""
    name: str
    path: str
    type: str
    size: Optional[int] = None

class DirectoryContents(BaseModel):
    """Model for directory browsing response."""
    current_path: str
    parent_path: str
    contents: List[DirectoryItem]

class PlotResponse(BaseModel):
    """Model for plot data response."""
    data: List[List[float]] 