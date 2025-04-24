import React, { useState } from 'react';
import { ProcessingParams, processImages } from '../api';

interface Props {
    onProcess: (message: string) => void;
    onError: (error: string) => void;
}

const defaultParams: ProcessingParams = {
    th_donut: 100,
    th_mask: 0.5,
    th_streak: 50,
    win_streak: 10,
    exp_donut: 2,
    exp_streak: 2
};

export function ParameterForm({ onProcess, onError }: Props) {
    const [params, setParams] = useState<ProcessingParams>(defaultParams);
    const [processing, setProcessing] = useState(false);

    async function handleSubmit(e: React.FormEvent) {
        e.preventDefault();
        try {
            setProcessing(true);
            const result = await processImages(params);
            onProcess(result.message);
        } catch (error) {
            onError(error instanceof Error ? error.message : 'Failed to process images');
        } finally {
            setProcessing(false);
        }
    }

    function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
        const { name, value } = e.target;
        setParams(prev => ({
            ...prev,
            [name]: name === 'th_mask' ? parseFloat(value) : parseInt(value, 10)
        }));
    }

    return (
        <>
            <form onSubmit={handleSubmit} className="parameter-form">
                <div className="parameter-group">
                    <h3>Donut Detection</h3>
                    <div className="parameter">
                        <label htmlFor="th_donut">Threshold:</label>
                        <input
                            type="number"
                            id="th_donut"
                            name="th_donut"
                            value={params.th_donut}
                            onChange={handleChange}
                            min="0"
                        />
                    </div>
                    <div className="parameter">
                        <label htmlFor="exp_donut">Exponent:</label>
                        <input
                            type="number"
                            id="exp_donut"
                            name="exp_donut"
                            value={params.exp_donut}
                            onChange={handleChange}
                            min="0"
                            step="1"
                        />
                    </div>
                </div>

                <div className="parameter-group">
                    <h3>Ring Mask</h3>
                    <div className="parameter">
                        <label htmlFor="th_mask">Threshold:</label>
                        <input
                            type="number"
                            id="th_mask"
                            name="th_mask"
                            value={params.th_mask}
                            onChange={handleChange}
                            min="0"
                            max="1"
                            step="0.1"
                        />
                    </div>
                </div>

                <div className="parameter-group">
                    <h3>Streak Detection</h3>
                    <div className="parameter">
                        <label htmlFor="th_streak">Threshold:</label>
                        <input
                            type="number"
                            id="th_streak"
                            name="th_streak"
                            value={params.th_streak}
                            onChange={handleChange}
                            min="0"
                        />
                    </div>
                    <div className="parameter">
                        <label htmlFor="win_streak">Window Size:</label>
                        <input
                            type="number"
                            id="win_streak"
                            name="win_streak"
                            value={params.win_streak}
                            onChange={handleChange}
                            min="1"
                        />
                    </div>
                    <div className="parameter">
                        <label htmlFor="exp_streak">Exponent:</label>
                        <input
                            type="number"
                            id="exp_streak"
                            name="exp_streak"
                            value={params.exp_streak}
                            onChange={handleChange}
                            min="0"
                            step="1"
                        />
                    </div>
                </div>

                <button type="submit" disabled={processing}>
                    {processing ? 'Processing...' : 'Process Images'}
                </button>
            </form>

            <style jsx>{`
                .parameter-form {
                    display: flex;
                    flex-direction: column;
                    gap: 2rem;
                    padding: 1rem;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                }

                .parameter-group {
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                }

                .parameter-group h3 {
                    margin: 0;
                    color: #333;
                    font-size: 1.1em;
                }

                .parameter {
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                }

                label {
                    min-width: 100px;
                    color: #666;
                }

                input {
                    width: 100px;
                    padding: 0.5rem;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                }

                button {
                    padding: 0.75rem;
                    background-color: #007bff;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    transition: background-color 0.2s;
                }

                button:hover:not(:disabled) {
                    background-color: #0056b3;
                }

                button:disabled {
                    background-color: #ccc;
                    cursor: not-allowed;
                }
            `}</style>
        </>
    );
} 