// Stock Details Page - Live Data & Interactive Charts
// Updates every 10 seconds for real-time analysis

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.querySelector('.sidebar-overlay');
    sidebar.classList.toggle('active');
    overlay.classList.toggle('active');
}

// Main function to fetch all stock data
async function fetchStockData() {
    await loadStockData();
}

// Load basic stock data
async function loadStockData() {
    try {
        const response = await fetch(`/api/stock_data/${currentSymbol}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            console.error('Error loading stock data:', data.error);
            return;
        }

        // Update stock name
        document.getElementById('stockName').textContent = data.name;
        
        // Update price display
        const priceChange = data.change >= 0 ? 'positive' : 'negative';
        const priceIcon = data.change >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';
        
        document.getElementById('stockPrice').innerHTML = `
            <div class="stock-price ${priceChange}">₹${data.current_price.toFixed(2)}</div>
            <div class="price-change ${priceChange}">
                <i class="fas ${priceIcon}"></i>
                ₹${Math.abs(data.change).toFixed(2)} (${Math.abs(data.change_percent).toFixed(2)}%)
            </div>
        `;

        // Update stock details section with grid layout (with safe null checks)
        const week52High = data['52_week_high'] || data.week_52_high;
        const week52Low = data['52_week_low'] || data.week_52_low;
        const dayHigh = data.high || data.day_high;
        const dayLow = data.low || data.day_low;
        
        document.getElementById('stockDetails').innerHTML = `
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Symbol</div>
                    <div class="info-value">${data.symbol || 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Current Price</div>
                    <div class="info-value">₹${data.current_price ? data.current_price.toFixed(2) : 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Day High</div>
                    <div class="info-value">₹${dayHigh ? dayHigh.toFixed(2) : 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Day Low</div>
                    <div class="info-value">₹${dayLow ? dayLow.toFixed(2) : 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">52 Week High</div>
                    <div class="info-value">₹${week52High ? week52High.toFixed(2) : 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">52 Week Low</div>
                    <div class="info-value">₹${week52Low ? week52Low.toFixed(2) : 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Volume</div>
                    <div class="info-value">${data.volume ? data.volume.toLocaleString() : 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">P/E Ratio</div>
                    <div class="info-value">${data.pe_ratio ? data.pe_ratio.toFixed(2) : 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Market Cap</div>
                    <div class="info-value">${data.market_cap ? '₹' + (data.market_cap / 10000000).toFixed(2) + ' Cr' : 'N/A'}</div>
                </div>
            </div>
        `;

        // Load other data sections
        loadHistoricalData(currentSymbol);
        loadTechnicalIndicators(currentSymbol);
        loadCAPMAnalysis(currentSymbol);
        loadAIInsights(currentSymbol);
    } catch (error) {
        console.error('Error in loadStockData:', error.message || error);
        // Show error message to user
        document.getElementById('stockDetails').innerHTML = `
            <div class="alert alert-danger">
                <strong>Error:</strong> Unable to load stock data. Please try again.
            </div>
        `;
    }
}

// Load historical data and predictions chart
async function loadHistoricalData(symbol) {
    try {
        const response = await fetch(`/api/historical/${symbol}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Use 'close' field from API (not 'prices')
        const prices = data.close || data.prices || [];
        const dates = data.dates || [];
        
        console.log('Historical data received:', {
            datesCount: dates.length,
            pricesCount: prices.length,
            hasOpen: !!data.open,
            hasHigh: !!data.high,
            hasLow: !!data.low
        });
        
        if (!data.error && dates.length > 0 && prices.length > 0) {
            // Get AI predictions
            const predResponse = await fetch(`/api/ai_insights/${symbol}`);
            const predData = await predResponse.json();
            
            const traces = [{
                x: dates,
                y: prices,
                type: 'scatter',
                mode: 'lines',
                name: 'Historical Price',
                line: {color: '#2563eb', width: 2},
                fill: 'tozeroy',
                fillcolor: 'rgba(37, 99, 235, 0.1)'
            }];
            
            // Add future prediction as a single point if available
            if (!predData.error && predData.prediction) {
                // Extract predicted price from prediction text
                const predMatch = predData.prediction.match(/₹([\d.]+)/);
                if (predMatch) {
                    const predPrice = parseFloat(predMatch[1]);
                    // Create future date (7 days from last date)
                    const lastDate = new Date(dates[dates.length - 1]);
                    lastDate.setDate(lastDate.getDate() + 7);
                    const futureDate = lastDate.toISOString().split('T')[0];
                    
                    traces.push({
                        x: [dates[dates.length - 1], futureDate],
                        y: [prices[prices.length - 1], predPrice],
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: '7-Day AI Forecast',
                        line: {color: '#10b981', width: 3, dash: 'dash'},
                        marker: {size: 10, symbol: 'star'}
                    });
                }
            }
            
            await Plotly.newPlot('priceChart', traces, {
                margin: {t: 10, r: 10, b: 40, l: 60},
                xaxis: {title: 'Date'},
                yaxis: {title: 'Price (₹)'},
                height: 450,
                hovermode: 'x unified',
                showlegend: true
            }, {responsive: true});
            console.log('Price chart rendered successfully');

            // Load candlestick chart using direct fields (not nested candlestick object)
            if (data.open && data.high && data.low && dates.length > 0) {
                await Plotly.newPlot('candlestickChart', [{
                    x: dates,
                    open: data.open,
                    high: data.high,
                    low: data.low,
                    close: prices,
                    type: 'candlestick',
                    increasing: {line: {color: '#10b981'}},
                    decreasing: {line: {color: '#ef4444'}}
                }], {
                    margin: {t: 10, r: 10, b: 40, l: 60},
                    xaxis: {title: 'Date'},
                    yaxis: {title: 'Price (₹)'},
                    height: 450
                }, {responsive: true});
                console.log('Candlestick chart rendered successfully');
            } else {
                // Show message if candlestick data not available
                document.getElementById('candlestickChart').innerHTML = `
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle me-2"></i>Candlestick data not available for this stock
                    </div>
                `;
            }
        } else {
            // No price data available
            document.getElementById('priceChart').innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>Price chart data not available
                </div>
            `;
        }
    } catch (error) {
        console.error('Error in loadHistoricalData:', error);
        document.getElementById('priceChart').innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-times-circle me-2"></i>Failed to load price chart
            </div>
        `;
    }
}

// Load technical indicators (RSI, MACD)
async function loadTechnicalIndicators(symbol) {
    try {
        const response = await fetch(`/api/technical/${symbol}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            // Show fallback message for technical indicators
            document.getElementById('rsiChart').innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>RSI data not available
                </div>
            `;
            document.getElementById('macdChart').innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>MACD data not available
                </div>
            `;
            return;
        }
        
        if (data.rsi && data.rsi.values && data.rsi.values.length > 0) {
            console.log('Rendering RSI chart with', data.rsi.values.length, 'data points');
            if (typeof Plotly === 'undefined') {
                console.error('Plotly is not defined!');
                return;
            }
            try {
                await Plotly.newPlot('rsiChart', [{
                x: data.rsi.dates,
                y: data.rsi.values,
                type: 'scatter',
                mode: 'lines',
                name: 'RSI',
                line: {color: '#8b5cf6', width: 2}
            }], {
                margin: {t: 10, r: 10, b: 40, l: 60},
                xaxis: {title: 'Date'},
                yaxis: {title: 'RSI', range: [0, 100]},
                shapes: [
                    {type: 'line', x0: data.rsi.dates[0], x1: data.rsi.dates[data.rsi.dates.length-1], y0: 70, y1: 70, line: {color: 'red', dash: 'dash', width: 1}},
                    {type: 'line', x0: data.rsi.dates[0], x1: data.rsi.dates[data.rsi.dates.length-1], y0: 30, y1: 30, line: {color: 'green', dash: 'dash', width: 1}}
                ],
                height: 300
            }, {responsive: true});
                console.log('RSI chart rendered successfully');
            } catch (error) {
                console.error('Error rendering RSI chart:', error);
                document.getElementById('rsiChart').innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-times-circle me-2"></i>Failed to render RSI chart
                    </div>
                `;
            }
        } else {
            console.log('RSI data not available or empty');
            document.getElementById('rsiChart').innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i>RSI indicator not available for this stock
                </div>
            `;
        }

        if (data.macd && data.macd.macd && data.macd.macd.length > 0) {
            console.log('Rendering MACD chart with', data.macd.macd.length, 'data points');
            if (typeof Plotly === 'undefined') {
                console.error('Plotly is not defined!');
                return;
            }
            try {
                await Plotly.newPlot('macdChart', [
                {x: data.macd.dates, y: data.macd.macd, type: 'scatter', mode: 'lines', name: 'MACD', line: {color: '#2563eb', width: 2}},
                {x: data.macd.dates, y: data.macd.signal, type: 'scatter', mode: 'lines', name: 'Signal', line: {color: '#f59e0b', width: 2}},
                {x: data.macd.dates, y: data.macd.histogram, type: 'bar', name: 'Histogram', marker: {color: '#10b981'}}
            ], {
                margin: {t: 10, r: 10, b: 40, l: 60},
                xaxis: {title: 'Date'},
                yaxis: {title: 'MACD'},
                height: 300
            }, {responsive: true});
                console.log('MACD chart rendered successfully');
            } catch (error) {
                console.error('Error rendering MACD chart:', error);
                document.getElementById('macdChart').innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-times-circle me-2"></i>Failed to render MACD chart
                    </div>
                `;
            }
        } else{
            console.log('MACD data not available or empty');
            document.getElementById('macdChart').innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i>MACD indicator not available for this stock
                </div>
            `;
        }
    } catch (error) {
        console.error('Error in loadTechnicalIndicators:', error);
    }
}

// Load CAPM analysis
async function loadCAPMAnalysis(symbol) {
    try {
        const response = await fetch(`/api/capm/${symbol}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (!data.error) {
            document.getElementById('capmAnalysis').innerHTML = `
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Beta</div>
                        <div class="info-value">${data.beta ? data.beta.toFixed(3) : 'N/A'}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Alpha</div>
                        <div class="info-value">${data.alpha ? data.alpha.toFixed(3) : 'N/A'}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Expected Return</div>
                        <div class="info-value">${data.expected_return ? data.expected_return.toFixed(2) + '%' : 'N/A'}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Market Return</div>
                        <div class="info-value">${data.market_return ? data.market_return.toFixed(2) + '%' : 'N/A'}</div>
                    </div>
                </div>
            `;

            if (data.sml_data) {
                Plotly.newPlot('smlChart', [
                    {x: data.sml_data.betas, y: data.sml_data.expected_returns, type: 'scatter', mode: 'lines', name: 'SML', line: {color: '#2563eb', width: 2}},
                    {x: [data.beta], y: [data.expected_return], type: 'scatter', mode: 'markers', name: symbol, marker: {size: 12, color: '#ef4444'}}
                ], {
                    margin: {t: 10, r: 10, b: 40, l: 60},
                    xaxis: {title: 'Beta'},
                    yaxis: {title: 'Expected Return (%)'},
                    height: 400
                }, {responsive: true});
            }
        }
    } catch (error) {
        console.error('Error in loadCAPMAnalysis:', error);
    }
}

// Load AI insights with fallback for undefined values
async function loadAIInsights(symbol) {
    try {
        const response = await fetch(`/api/ai_insights/${symbol}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Provide fallback values for undefined fields
        const sentiment = data.sentiment || 'Neutral';
        const confidence = data.confidence || 50;
        const prediction = data.prediction || 'Not enough data';
        const recommendation = data.recommendation || 'Hold';
        
        const sentimentClass = sentiment === 'Positive' ? 'text-success' : 
                               sentiment === 'Negative' ? 'text-danger' : 'text-warning';
        
        document.getElementById('aiInsights').innerHTML = `
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Sentiment Analysis</div>
                    <div class="info-value ${sentimentClass}">
                        <i class="fas fa-${sentiment === 'Positive' ? 'smile' : sentiment === 'Negative' ? 'frown' : 'meh'}"></i>
                        ${sentiment}
                    </div>
                </div>
                <div class="info-item">
                    <div class="info-label">AI Confidence</div>
                    <div class="info-value">${confidence}%</div>
                    <div class="progress mt-2" style="height: 8px;">
                        <div class="progress-bar bg-success" role="progressbar" style="width: ${confidence}%"></div>
                    </div>
                </div>
                <div class="info-item">
                    <div class="info-label">Price Prediction</div>
                    <div class="info-value">${prediction}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Trading Recommendation</div>
                    <div class="info-value">
                        <span class="badge ${recommendation === 'Buy' ? 'bg-success' : recommendation === 'Sell' ? 'bg-danger' : 'bg-warning'} fs-6">
                            ${recommendation}
                        </span>
                    </div>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Error in loadAIInsights:', error);
        // Show fallback UI on error
        document.getElementById('aiInsights').innerHTML = `
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Sentiment Analysis</div>
                    <div class="info-value text-warning">Neutral</div>
                </div>
                <div class="info-item">
                    <div class="info-label">AI Confidence</div>
                    <div class="info-value">50%</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Price Prediction</div>
                    <div class="info-value">Not enough data</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Trading Recommendation</div>
                    <div class="info-value"><span class="badge bg-warning fs-6">Hold</span></div>
                </div>
            </div>
        `;
    }
}

// Add stock to watchlist
async function addToWatchlist() {
    try {
        const response = await fetch('/api/watchlist', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({symbol: currentSymbol})
        });
        
        const data = await response.json();
        alert(data.message || data.error);
    } catch (error) {
        console.error('Error adding to watchlist:', error);
        alert('Error adding stock to watchlist');
    }
}

// Initialize page - load all data immediately
fetchStockData();

// Set up auto-refresh every 10 seconds for live updates
setInterval(fetchStockData, 10000);
