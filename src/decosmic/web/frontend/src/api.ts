/**
 * API client for interacting with the XRD Decosmic backend.
 */

const API_BASE = 'http://localhost:8000';

export interface DirectoryContents {
    current_path: string;
    parent_path: string;
    contents: {
        name: string;
        path: string;
        type: 'directory' | 'file';
        size: number | null;
    }[];
}

export interface ProcessingParams {
    th_donut: number;
    th_mask: number;
    th_streak: number;
    win_streak: number;
    exp_donut: number;
    exp_streak: number;
}

export interface ProcessingResponse {
    status: string;
    message: string;
}

/**
 * Browse directory contents on the server.
 */
export async function browseDirectory(path: string): Promise<DirectoryContents> {
    const response = await fetch(`${API_BASE}/browse/${encodeURIComponent(path)}`);
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to browse directory');
    }
    return response.json();
}

/**
 * Load data from a server path.
 */
export async function loadFromPath(path: string): Promise<ProcessingResponse> {
    const response = await fetch(`${API_BASE}/load-path`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path })
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to load data');
    }
    return response.json();
}

/**
 * Upload a file (for small datasets only).
 */
export async function uploadFile(file: File): Promise<ProcessingResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to upload file');
    }
    return response.json();
}

/**
 * Process images with given parameters.
 */
export async function processImages(params: ProcessingParams): Promise<ProcessingResponse> {
    const response = await fetch(`${API_BASE}/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to process images');
    }
    return response.json();
}

/**
 * Get plot data for visualization.
 */
export async function getPlotData(plotType: string): Promise<number[][]> {
    const response = await fetch(`${API_BASE}/plot/${plotType}`);
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to get plot data');
    }
    const result = await response.json();
    return result.data;
} 