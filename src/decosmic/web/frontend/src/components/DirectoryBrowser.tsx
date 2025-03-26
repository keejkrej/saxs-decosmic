import React, { useState, useEffect } from 'react';
import { browseDirectory, loadFromPath, DirectoryContents } from '../api';

interface Props {
    onLoad: (message: string) => void;
    onError: (error: string) => void;
}

export function DirectoryBrowser({ onLoad, onError }: Props) {
    const [currentPath, setCurrentPath] = useState('/');
    const [contents, setContents] = useState<DirectoryContents | null>(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        loadDirectory(currentPath);
    }, [currentPath]);

    async function loadDirectory(path: string) {
        try {
            setLoading(true);
            const result = await browseDirectory(path);
            setContents(result);
        } catch (error) {
            onError(error instanceof Error ? error.message : 'Failed to browse directory');
        } finally {
            setLoading(false);
        }
    }

    async function handleFileClick(path: string) {
        try {
            setLoading(true);
            const result = await loadFromPath(path);
            onLoad(result.message);
        } catch (error) {
            onError(error instanceof Error ? error.message : 'Failed to load file');
        } finally {
            setLoading(false);
        }
    }

    function handleDirectoryClick(path: string) {
        setCurrentPath(path);
    }

    if (loading) {
        return <div className="loading">Loading...</div>;
    }

    return (
        <>
            <div className="directory-browser">
                <div className="path-bar">
                    <button 
                        onClick={() => contents?.parent_path && setCurrentPath(contents.parent_path)}
                        disabled={!contents?.parent_path || contents.parent_path === contents.current_path}
                    >
                        ⬆️ Up
                    </button>
                    <span className="current-path">{currentPath}</span>
                </div>
                
                <div className="contents">
                    {contents?.contents.map((item) => (
                        <div 
                            key={item.path}
                            className={`item ${item.type}`}
                            onClick={() => item.type === 'directory' 
                                ? handleDirectoryClick(item.path)
                                : handleFileClick(item.path)
                            }
                        >
                            <span className="icon">
                                {item.type === 'directory' ? '📁' : '📄'}
                            </span>
                            <span className="name">{item.name}</span>
                            {item.size !== null && (
                                <span className="size">
                                    {(item.size / 1024 / 1024).toFixed(2)} MB
                                </span>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            <style jsx>{`
                .directory-browser {
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    padding: 1rem;
                    max-height: 400px;
                    display: flex;
                    flex-direction: column;
                }

                .path-bar {
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                    padding-bottom: 1rem;
                    border-bottom: 1px solid #eee;
                }

                .current-path {
                    font-family: monospace;
                    color: #666;
                }

                .contents {
                    overflow-y: auto;
                    flex-grow: 1;
                }

                .item {
                    display: flex;
                    align-items: center;
                    padding: 0.5rem;
                    cursor: pointer;
                    transition: background-color 0.2s;
                }

                .item:hover {
                    background-color: #f5f5f5;
                }

                .icon {
                    margin-right: 0.5rem;
                }

                .name {
                    flex-grow: 1;
                }

                .size {
                    color: #666;
                    font-size: 0.9em;
                }

                .loading {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    height: 200px;
                    color: #666;
                }
            `}</style>
        </>
    );
} 