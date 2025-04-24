from fastapi import APIRouter, UploadFile, Depends
from pathlib import Path
import numpy as np

from xrd_decosmic.web.backend.models.schemas import (
    ProcessingResponse,
    PathRequest,
    DirectoryContents,
    PlotResponse
)
from xrd_decosmic.web.backend.services.processor_service import ProcessorService
from xrd_decosmic.web.backend.services.filesystem_service import FileSystemService
from xrd_decosmic.core import ProcessingParams

router = APIRouter()

def get_processor_service() -> ProcessorService:
    """Dependency injection for ProcessorService."""
    return ProcessorService()

@router.post("/load-path", response_model=ProcessingResponse)
async def load_from_path(
    request: PathRequest,
    processor_service: ProcessorService = Depends(get_processor_service)
):
    """Load XRD image series from an existing path."""
    message, _ = processor_service.load_from_path(request.path)
    return ProcessingResponse(status="success", message=message)

@router.post("/upload", response_model=ProcessingResponse)
async def upload_file(
    file: UploadFile,
    processor_service: ProcessorService = Depends(get_processor_service)
):
    """Upload XRD image file (for small datasets only)."""
    # Create temporary directory if it doesn't exist
    tmp_dir = Path("/tmp/xrd-decosmic")
    tmp_dir.mkdir(exist_ok=True)
    
    # Save uploaded file temporarily
    file_path = tmp_dir / file.filename
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    
    message, _ = processor_service.load_from_path(str(file_path))
    return ProcessingResponse(status="success", message=message)

@router.get("/browse/{directory:path}", response_model=DirectoryContents)
async def browse_directory(directory: str = "/"):
    """Browse server directories for XRD image files."""
    return FileSystemService.browse_directory(directory)

@router.post("/process", response_model=ProcessingResponse)
async def process_images(
    params: ProcessingParams,
    processor_service: ProcessorService = Depends(get_processor_service)
):
    """Process images with given parameters."""
    message = processor_service.process_images(params)
    return ProcessingResponse(status="success", message=message)

@router.get("/plot/{plot_type}", response_model=PlotResponse)
async def get_plot_data(
    plot_type: str,
    processor_service: ProcessorService = Depends(get_processor_service)
):
    """Get plot data for visualization."""
    data = processor_service.get_plot_data(plot_type)
    return PlotResponse(data=data.tolist()) 