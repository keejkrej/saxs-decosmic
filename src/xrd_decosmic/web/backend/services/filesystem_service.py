from pathlib import Path
from typing import List, Dict, Any
from fastapi import HTTPException

from ..models.schemas import DirectoryItem, DirectoryContents

class FileSystemService:
    ALLOWED_ROOTS = ["/data", "/home", "/tmp"]

    @staticmethod
    def browse_directory(directory: str = "/") -> DirectoryContents:
        """Browse server directories for XRD image files."""
        try:
            # Convert to absolute path and resolve any symlinks
            path = Path(directory).resolve()
            
            # Basic security check
            if not any(str(path).startswith(root) for root in FileSystemService.ALLOWED_ROOTS):
                raise ValueError("Access to this directory is not allowed")
            
            # Get directory contents
            contents: List[DirectoryItem] = []
            try:
                for item in path.iterdir():
                    try:
                        is_dir = item.is_dir()
                        contents.append(DirectoryItem(
                            name=item.name,
                            path=str(item),
                            type="directory" if is_dir else "file",
                            size=None if is_dir else item.stat().st_size
                        ))
                    except (PermissionError, OSError):
                        continue  # Skip items we can't access
            except PermissionError:
                raise ValueError("Permission denied")
                
            # Sort contents: directories first, then files, both alphabetically
            contents.sort(key=lambda x: (x.type == "file", x.name))
            
            return DirectoryContents(
                current_path=str(path),
                parent_path=str(path.parent),
                contents=contents
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) 