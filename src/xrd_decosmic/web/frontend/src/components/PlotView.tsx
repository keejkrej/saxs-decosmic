import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import { getPlotData } from '../api';

interface Props {
    plotType: string;
    onError: (error: string) => void;
}

export function PlotView({ plotType, onError }: Props) {
    const [data, setData] = useState<number[][]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        async function fetchData() {
            try {
                setLoading(true);
                const result = await getPlotData(plotType);
                setData(result);
            } catch (error) {
                onError(error instanceof Error ? error.message : 'Failed to load plot data');
            } finally {
                setLoading(false);
            }
        }

        if (plotType) {
            fetchData();
        }
    }, [plotType, onError]);

    if (loading) {
        return <div className="loading">Loading plot data...</div>;
    }

    if (!data.length) {
        return <div className="no-data">No data available</div>;
    }

    return (
        <>
            <div className="plot-view">
                <Plot
                    data={[
                        {
                            z: data,
                            type: 'heatmap',
                            colorscale: 'Hot',
                            showscale: true
                        }
                    ]}
                    layout={{
                        width: 600,
                        height: 600,
                        title: plotType.charAt(0).toUpperCase() + plotType.slice(1),
                        margin: { t: 30, r: 30, b: 30, l: 30 },
                        xaxis: { scaleanchor: 'y' }
                    }}
                    config={{
                        responsive: true,
                        displayModeBar: true,
                        modeBarButtonsToRemove: ['lasso2d', 'select2d']
                    }}
                />
            </div>

            <style jsx>{`
                .plot-view {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    padding: 1rem;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    background: white;
                }

                .loading,
                .no-data {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    height: 400px;
                    color: #666;
                }
            `}</style>
        </>
    );
} 