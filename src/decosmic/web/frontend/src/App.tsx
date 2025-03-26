import React, { useState } from 'react';
import { DirectoryBrowser } from './components/DirectoryBrowser';
import { ParameterForm } from './components/ParameterForm';
import { PlotView } from './components/PlotView';

export function App() {
    const [message, setMessage] = useState<string>('');
    const [error, setError] = useState<string>('');
    const [plotType, setPlotType] = useState<string>('average');
    const [dataLoaded, setDataLoaded] = useState(false);

    function handleLoad(msg: string) {
        setMessage(msg);
        setError('');
        setDataLoaded(true);
    }

    function handleError(err: string) {
        setError(err);
        setMessage('');
    }

    function handleProcess(msg: string) {
        setMessage(msg);
        setError('');
    }

    return (
        <div className="app">
            <header>
                <h1>XRD Decosmic</h1>
                {message && <div className="message">{message}</div>}
                {error && <div className="error">{error}</div>}
            </header>

            <main>
                <section className="data-section">
                    <h2>Load Data</h2>
                    <DirectoryBrowser onLoad={handleLoad} onError={handleError} />
                </section>

                {dataLoaded && (
                    <>
                        <section className="parameters-section">
                            <h2>Processing Parameters</h2>
                            <ParameterForm onProcess={handleProcess} onError={handleError} />
                        </section>

                        <section className="plot-section">
                            <h2>Visualization</h2>
                            <div className="plot-controls">
                                <select 
                                    value={plotType} 
                                    onChange={(e) => setPlotType(e.target.value)}
                                >
                                    <option value="average">Average</option>
                                    <option value="clean">Clean</option>
                                    <option value="difference">Difference</option>
                                    <option value="mask">Ring Mask</option>
                                    <option value="donut">Donut</option>
                                    <option value="streak">Streak</option>
                                </select>
                            </div>
                            <PlotView plotType={plotType} onError={handleError} />
                        </section>
                    </>
                )}
            </main>

            <style jsx>{`
                .app {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 2rem;
                }

                header {
                    margin-bottom: 2rem;
                    text-align: center;
                }

                h1 {
                    font-size: 2.5rem;
                    color: #333;
                    margin-bottom: 1rem;
                }

                .message {
                    padding: 0.5rem 1rem;
                    background-color: #d4edda;
                    color: #155724;
                    border-radius: 4px;
                    margin-bottom: 1rem;
                }

                .error {
                    padding: 0.5rem 1rem;
                    background-color: #f8d7da;
                    color: #721c24;
                    border-radius: 4px;
                    margin-bottom: 1rem;
                }

                main {
                    display: grid;
                    gap: 2rem;
                }

                section {
                    background: #f8f9fa;
                    padding: 1.5rem;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }

                h2 {
                    margin-top: 0;
                    margin-bottom: 1rem;
                    color: #495057;
                    font-size: 1.5rem;
                }

                .plot-controls {
                    margin-bottom: 1rem;
                }

                select {
                    padding: 0.5rem;
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    background-color: white;
                    min-width: 200px;
                }

                @media (min-width: 768px) {
                    main {
                        grid-template-columns: repeat(2, 1fr);
                    }

                    .data-section {
                        grid-column: 1 / -1;
                    }

                    .plot-section {
                        grid-column: 1 / -1;
                    }
                }
            `}</style>
        </div>
    );
}
