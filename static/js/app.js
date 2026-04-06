/**
 * Multi-Asset Financial Forecaster — Frontend Logic
 */

const INR = new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 2 });

async function predict(assetKey) {
    const btn = document.querySelector(`.asset-card[data-asset="${assetKey}"] .btn-predict`);
    const loader = document.getElementById(`loader-${assetKey}`);
    const priceEl = document.getElementById(`price-${assetKey}`);

    btn.disabled = true;
    btn.textContent = 'Training...';
    loader.style.display = 'flex';

    try {
        const res = await fetch(`/api/predict/${assetKey}`);
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || 'Prediction failed');
        }
        const data = await res.json();

        // Update price
        priceEl.textContent = INR.format(data.current_price);

        // Show chart
        renderChart(data);
        renderMetrics(data);
        renderSummary(data);

        document.getElementById('chart-section').style.display = 'block';
        document.getElementById('chart-section').scrollIntoView({ behavior: 'smooth' });
    } catch (e) {
        alert(`Error: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Generate Forecast';
        loader.style.display = 'none';
    }
}

function renderChart(data) {
    const histTrace = {
        x: data.historical.dates,
        y: data.historical.prices,
        type: 'scatter',
        mode: 'lines',
        name: 'Historical',
        line: { color: '#3b82f6', width: 2 },
    };

    const forecastTrace = {
        x: data.forecast.dates,
        y: data.forecast.prices,
        type: 'scatter',
        mode: 'lines',
        name: 'Forecast (6M)',
        line: { color: '#10b981', width: 2, dash: 'dot' },
        fill: 'tozeroy',
        fillcolor: 'rgba(16,185,129,0.08)',
    };

    const layout = {
        title: { text: `${data.name} — Price Forecast`, font: { color: '#f1f5f9', size: 16 } },
        paper_bgcolor: '#1a2235',
        plot_bgcolor: '#111827',
        font: { color: '#94a3b8', family: 'Inter' },
        xaxis: {
            gridcolor: '#1e293b',
            title: 'Date',
        },
        yaxis: {
            gridcolor: '#1e293b',
            title: 'Price (₹)',
            tickformat: ',.0f',
        },
        legend: { orientation: 'h', y: -0.15 },
        margin: { t: 50, r: 30, b: 60, l: 80 },
        shapes: [{
            type: 'line',
            x0: data.historical.dates[data.historical.dates.length - 1],
            x1: data.historical.dates[data.historical.dates.length - 1],
            y0: 0, y1: 1, yref: 'paper',
            line: { color: '#f59e0b', width: 1, dash: 'dash' },
        }],
        annotations: [{
            x: data.historical.dates[data.historical.dates.length - 1],
            y: 1, yref: 'paper',
            text: 'Today',
            showarrow: false,
            font: { color: '#f59e0b', size: 11 },
            yanchor: 'bottom',
        }],
    };

    document.getElementById('chart-title').textContent = `${data.name} — Forecast`;
    Plotly.newPlot('chart-container', [histTrace, forecastTrace], layout, { responsive: true });
}

function renderMetrics(data) {
    const bar = document.getElementById('metrics-bar');
    bar.innerHTML = '';
    for (const [model, m] of Object.entries(data.metrics)) {
        const best = model === data.best_model ? ' ⭐' : '';
        bar.innerHTML += `<span class="metric-chip"><strong>${model}${best}</strong> RMSE: ₹${m.rmse.toLocaleString('en-IN')} | MAE: ₹${m.mae.toLocaleString('en-IN')}</span>`;
    }
}

function renderSummary(data) {
    const summary = document.getElementById('forecast-summary');
    const lastForecast = data.forecast.prices[data.forecast.prices.length - 1];
    const change = lastForecast - data.current_price;
    const changePct = ((change / data.current_price) * 100).toFixed(2);
    const direction = change >= 0 ? 'up' : 'down';
    const arrow = change >= 0 ? '▲' : '▼';

    summary.innerHTML = `
        <div class="summary-item">
            <div class="summary-label">Current Price</div>
            <div class="summary-value">${INR.format(data.current_price)}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">6-Month Forecast</div>
            <div class="summary-value ${direction}">${INR.format(lastForecast)}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Projected Change</div>
            <div class="summary-value ${direction}">${arrow} ${changePct}%</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Best Model</div>
            <div class="summary-value">${data.best_model}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">USD/INR Rate</div>
            <div class="summary-value">₹${data.usd_inr_rate}</div>
        </div>
    `;
}
