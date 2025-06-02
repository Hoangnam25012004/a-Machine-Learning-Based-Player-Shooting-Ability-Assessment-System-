import pandas as pd
import numpy as np
import json

# Read the data
df = pd.read_csv(r"D:\Central forward stat\events.csv")
ratings_df = pd.read_csv('player_shooting_ratings.csv')

def create_shooting_ability_target(row):
    """Create target shooting ability score (0-100) based on shot quality factors"""
    if row['event_type'] != 1:  # Only for shot attempts
        return np.nan
    
    score = 0
    
    # Shot placement scoring (40 points max)
    shot_place_scores = {
        3: 40, 4: 40, 5: 35, 11: 38, 12: 40, 13: 40, 7: 25,
        1: 15, 2: 10, 6: 5, 8: 8, 9: 8, 10: 5
    }
    score += shot_place_scores.get(row['shot_place'], 0)
    
    # Shot outcome scoring (25 points max)
    outcome_scores = {1: 25, 4: 20, 2: 5, 3: 10}
    score += outcome_scores.get(row['shot_outcome'], 0)
    
    # Location scoring (20 points max)
    location_scores = {
        3: 20, 13: 20, 14: 18, 9: 15, 11: 15, 10: 15, 12: 15,
        15: 10, 16: 8, 6: 5, 7: 6, 8: 6, 17: 3, 18: 2, 1: 8, 4: 7, 5: 7
    }
    score += location_scores.get(row['location'], 0)
    
    # Body part scoring (8 points max)
    bodypart_scores = {1: 8, 2: 8, 3: 6}
    score += bodypart_scores.get(row['bodypart'], 0)
    
    # Assist method bonus (4 points max)
    assist_scores = {0: 2, 1: 4, 2: 3, 3: 3, 4: 4}
    score += assist_scores.get(row['assist_method'], 0)
    
    # Situation bonus (3 points max)
    situation_scores = {1: 3, 2: 2, 3: 2, 4: 2}
    score += situation_scores.get(row['situation'], 0)
    
    return min(score, 100)

# Create shooting ability scores
df['shooting_ability_score'] = df.apply(create_shooting_ability_target, axis=1)

# Filter for shot attempts only
shot_data = df[df['event_type'] == 1].copy()
shot_data = shot_data.dropna(subset=['shooting_ability_score'])

print(f"Processing {len(shot_data)} shots...")

# Convert data to JSON for embedding
players_data = ratings_df.to_dict('records')
events_data = shot_data.fillna('').to_dict('records')

# Create the embedded HTML file
html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚽ Player Shooting Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .player-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .rating-circle {{
            width: 80px;
            height: 80px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: bold;
            color: white;
            margin: 0 auto;
        }}
        .rating-excellent {{ background: #28a745; }}
        .rating-good {{ background: #17a2b8; }}
        .rating-average {{ background: #ffc107; color: #000; }}
        .rating-poor {{ background: #dc3545; }}
        
        .search-container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 15px 10px;
        }}
        .player-list {{
            max-height: 500px;
            overflow-y: auto;
            padding: 10px 0;
        }}
        #playerProfile {{
            display: none;
        }}
        .shot-history {{
            max-height: 300px;
            overflow-y: auto;
        }}
        
        h1 {{
            font-size: 2.2rem;
            margin-bottom: 15px;
        }}
        .lead {{
            font-size: 1.1rem;
            margin-bottom: 25px;
        }}
        h5 {{
            font-size: 1.1rem;
            margin-bottom: 8px;
            margin-top: 15px;
        }}
        .btn-sm {{
            font-size: 0.85rem;
            padding: 6px 12px;
        }}
        .card-body {{
            padding: 15px;
        }}
        .mb-3 {{
            margin-bottom: 15px !important;
        }}
        .mt-4 {{
            margin-top: 10px !important;
        }}
        
        .player-list .btn {{
            margin-bottom: 4px;
            padding: 8px 12px;
            font-size: 0.9rem;
            text-align: left;
            width: 100%;
            border: 1px solid #dee2e6;
            background-color: white;
            color: #007bff;
            display: block;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        
        .player-list .btn:hover {{
            background-color: #f8f9fa;
            border-color: #007bff;
        }}
        
        .player-list .col-md-6 {{
            padding: 0 4px;
            margin-bottom: 0;
        }}
        
        .player-list .badge {{
            font-size: 0.75rem;
            padding: 3px 6px;
            margin-left: 8px;
            float: right;
        }}
        
        .player-list .row {{
            margin: 0 -4px;
        }}
        
        .container {{
            padding-left: 10px;
            padding-right: 10px;
        }}
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="#" onclick="showHome()">⚽ Player Shooting Analysis</a>
        </div>
    </nav>

    <div class="container mt-4">
        <!-- Home Page -->
        <div id="homePage">
            <div class="search-container">
                <div class="text-center mb-4">
                    <h1>🎯 Player Shooting Analysis</h1>
                    <p class="lead">Search for a player to view their shooting statistics and performance over time</p>
                </div>

                <div class="card">
                    <div class="card-body">
                        <div class="mb-3">
                            <label for="playerSearch" class="form-label">Search Player:</label>
                            <input type="text" class="form-control form-control-lg" id="playerSearch" 
                                   placeholder="Enter player name..." autocomplete="off">
                            <div id="searchResults" class="list-group mt-2" style="display: none;"></div>
                        </div>
                        <button onclick="searchPlayer()" class="btn btn-primary btn-lg w-100">View Player Profile</button>
                    </div>
                </div>

                <div class="mt-4">
                    <h5>Available Players (Top 20 by Rating):</h5>
                    <div id="playerList" class="player-list">
                        <!-- Will be populated by JavaScript -->
                    </div>
                </div>
            </div>
        </div>

        <!-- Player Profile Page -->
        <div id="playerProfile">
            <div class="row">
                <div class="col-12">
                    <button onclick="showHome()" class="btn btn-secondary mb-3">← Back to Search</button>
                </div>
            </div>

            <div class="player-card">
                <div class="row align-items-center">
                    <div class="col-md-8">
                        <h1 id="playerName">Player Name</h1>
                        <p class="mb-0">Complete Shooting Performance Analysis</p>
                    </div>
                    <div class="col-md-4 text-center">
                        <div id="ratingCircle" class="rating-circle">
                            0.0
                        </div>
                        <small>Overall Rating</small>
                    </div>
                </div>
            </div>

            <div class="row" id="statsCards">
                <!-- Stats cards will be populated by JavaScript -->
            </div>

            <div class="row mt-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>📈 Shooting Ability Over Time</h5>
                        </div>
                        <div class="card-body">
                            <div id="chart" style="width:100%;height:500px;"></div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5>📊 Performance Metrics</h5>
                        </div>
                        <div class="card-body" id="performanceMetrics">
                            <!-- Performance metrics will be populated by JavaScript -->
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5>🎯 Shot History</h5>
                        </div>
                        <div class="card-body shot-history" id="shotHistory">
                            <!-- Shot history will be populated by JavaScript -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Embedded data - no need to load external files
        const playersData = {json.dumps(players_data, indent=2)};
        const eventsData = {json.dumps(events_data, indent=2)};
        
        console.log(`Loaded ${{playersData.length}} players and ${{eventsData.length}} events`);

        // Initialize the page
        document.addEventListener('DOMContentLoaded', function() {{
            loadPlayerList();
        }});

        function loadPlayerList() {{
            const playerList = document.getElementById('playerList');
            
            // Show top 20 players by rating
            const topPlayers = playersData.slice(0, 20);
            
            let html = '<div class="row">';
            topPlayers.forEach((player, index) => {{
                html += `
                    <div class="col-md-6">
                        <button onclick="showPlayerProfile('${{player.Player}}')" 
                                class="btn btn-outline-primary btn-sm">
                            ${{index + 1}}. ${{player.Player}}
                            <span class="badge bg-primary">${{player.Overall_Shooting_Rating}}</span>
                        </button>
                    </div>
                `;
            }});
            html += '</div>';
            
            playerList.innerHTML = html;
        }}

        function searchPlayer() {{
            const playerName = document.getElementById('playerSearch').value.trim();
            if (playerName) {{
                showPlayerProfile(playerName);
            }}
        }}

        function showPlayerProfile(playerName) {{
            const player = playersData.find(p => p.Player.toLowerCase() === playerName.toLowerCase());
            
            if (!player) {{
                alert(`Player "${{playerName}}" not found. Please check the spelling or select from the available players.`);
                return;
            }}

            // Get player shot data
            const playerShots = eventsData.filter(shot => 
                shot.player && shot.player.toLowerCase() === playerName.toLowerCase()
            );

            if (playerShots.length === 0) {{
                alert(`No shot data found for "${{playerName}}".`);
                return;
            }}

            // Update player name
            document.getElementById('playerName').textContent = player.Player;

            // Update rating circle
            const ratingCircle = document.getElementById('ratingCircle');
            const rating = player.Overall_Shooting_Rating;
            ratingCircle.textContent = rating.toFixed(1);
            
            // Set rating color
            ratingCircle.className = 'rating-circle ';
            if (rating >= 80) ratingCircle.className += 'rating-excellent';
            else if (rating >= 65) ratingCircle.className += 'rating-good';
            else if (rating >= 50) ratingCircle.className += 'rating-average';
            else ratingCircle.className += 'rating-poor';

            // Update stats cards
            updateStatsCards(player);

            // Update performance metrics
            updatePerformanceMetrics(player);

            // Create chart
            createChart(playerShots, player.Player);

            // Update shot history
            updateShotHistory(playerShots);

            // Show player profile
            document.getElementById('homePage').style.display = 'none';
            document.getElementById('playerProfile').style.display = 'block';
        }}

        function updateStatsCards(player) {{
            const statsCards = document.getElementById('statsCards');
            statsCards.innerHTML = `
                <div class="col-md-3">
                    <div class="stat-card text-center">
                        <h3 class="text-primary">${{player.Total_Shots}}</h3>
                        <p class="mb-0">Total Shots</p>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card text-center">
                        <h3 class="text-success">${{player.Goals}}</h3>
                        <p class="mb-0">Goals Scored</p>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card text-center">
                        <h3 class="text-info">${{player['Goal_Rate_%'].toFixed(1)}}%</h3>
                        <p class="mb-0">Conversion Rate</p>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card text-center">
                        <h3 class="text-warning">${{player.Avg_Shot_Quality.toFixed(1)}}</h3>
                        <p class="mb-0">Avg Shot Quality</p>
                    </div>
                </div>
            `;
        }}

        function updatePerformanceMetrics(player) {{
            const performanceMetrics = document.getElementById('performanceMetrics');
            performanceMetrics.innerHTML = `
                <div class="row">
                    <div class="col-6">
                        <strong>Shot Consistency:</strong><br>
                        <span class="text-muted">${{player.Shot_Consistency.toFixed(1)}} (lower = more consistent)</span>
                    </div>
                    <div class="col-6">
                        <strong>Best Shot:</strong><br>
                        <span class="text-success">${{player.Best_Shot.toFixed(1)}}/100</span>
                    </div>
                </div>
                <div class="row mt-3">
                    <div class="col-6">
                        <strong>Worst Shot:</strong><br>
                        <span class="text-danger">${{player.Worst_Shot.toFixed(1)}}/100</span>
                    </div>
                    <div class="col-6">
                        <strong>Range:</strong><br>
                        <span class="text-info">${{(player.Best_Shot - player.Worst_Shot).toFixed(1)}} points</span>
                    </div>
                </div>
            `;
        }}

        function calculateShootingAbility(shot) {{
            let score = 0;
            
            // Shot placement scoring (40 points max)
            const shotPlaceScores = {{
                3: 40, 4: 40, 5: 35, 11: 38, 12: 40, 13: 40, 7: 25,
                1: 15, 2: 10, 6: 5, 8: 8, 9: 8, 10: 5
            }};
            score += shotPlaceScores[shot.shot_place] || 0;
            
            // Shot outcome scoring (25 points max)
            const outcomeScores = {{1: 25, 4: 20, 2: 5, 3: 10}};
            score += outcomeScores[shot.shot_outcome] || 0;
            
            // Location scoring (20 points max)
            const locationScores = {{
                3: 20, 13: 20, 14: 18, 9: 15, 11: 15, 10: 15, 12: 15,
                15: 10, 16: 8, 6: 5, 7: 6, 8: 6, 17: 3, 18: 2, 1: 8, 4: 7, 5: 7
            }};
            score += locationScores[shot.location] || 0;
            
            // Body part scoring (8 points max)
            const bodypartScores = {{1: 8, 2: 8, 3: 6}};
            score += bodypartScores[shot.bodypart] || 0;
            
            // Assist method bonus (4 points max)
            const assistScores = {{0: 2, 1: 4, 2: 3, 3: 3, 4: 4}};
            score += assistScores[shot.assist_method] || 0;
            
            // Situation bonus (3 points max)
            const situationScores = {{1: 3, 2: 2, 3: 2, 4: 2}};
            score += situationScores[shot.situation] || 0;
            
            return Math.min(score, 100);
        }}

        function createChart(playerShots, playerName) {{
            if (playerShots.length === 0) {{
                document.getElementById('chart').innerHTML = '<p class="text-center text-muted">No shot data available for this player.</p>';
                return;
            }}

            // Calculate shooting ability for each shot
            const shotsWithAbility = playerShots.map(shot => ({{
                ...shot,
                shooting_ability: calculateShootingAbility(shot),
                time_minutes: parseFloat(shot.time) || 0
            }})).sort((a, b) => a.time_minutes - b.time_minutes);

            const trace = {{
                x: shotsWithAbility.map(shot => shot.time_minutes),
                y: shotsWithAbility.map(shot => shot.shooting_ability),
                mode: 'markers+lines',
                type: 'scatter',
                name: 'Shooting Ability',
                marker: {{
                    size: 8,
                    color: shotsWithAbility.map(shot => parseInt(shot.is_goal) || 0),
                    colorscale: [[0, 'red'], [1, 'green']],
                    showscale: true,
                    colorbar: {{title: "Goal (1) / No Goal (0)"}}
                }},
                line: {{width: 2, color: 'blue'}},
                hovertemplate: '<b>Time:</b> %{{x}} min<br>' +
                              '<b>Shooting Ability:</b> %{{y}}<br>' +
                              '<b>Goal:</b> %{{marker.color}}<br>' +
                              '<extra></extra>'
            }};

            const avgAbility = shotsWithAbility.reduce((sum, shot) => sum + shot.shooting_ability, 0) / shotsWithAbility.length;

            const layout = {{
                title: `${{playerName}} - Shooting Ability Over Time`,
                xaxis: {{title: 'Time (minutes)'}},
                yaxis: {{title: 'Shooting Ability Score'}},
                height: 500,
                shapes: [{{
                    type: 'line',
                    x0: Math.min(...shotsWithAbility.map(s => s.time_minutes)),
                    x1: Math.max(...shotsWithAbility.map(s => s.time_minutes)),
                    y0: avgAbility,
                    y1: avgAbility,
                    line: {{
                        color: 'orange',
                        width: 2,
                        dash: 'dash'
                    }}
                }}],
                annotations: [{{
                    x: Math.max(...shotsWithAbility.map(s => s.time_minutes)) * 0.8,
                    y: avgAbility + 5,
                    text: `Average: ${{avgAbility.toFixed(1)}}`,
                    showarrow: false,
                    font: {{color: 'orange'}}
                }}]
            }};

            Plotly.newPlot('chart', [trace], layout);
        }}

        function updateShotHistory(playerShots) {{
            const shotHistory = document.getElementById('shotHistory');
            
            if (playerShots.length === 0) {{
                shotHistory.innerHTML = '<p class="text-muted">No shot data available.</p>';
                return;
            }}

            // Show all shots
            const allShots = playerShots.reverse();
            
            let html = '';
            allShots.forEach(shot => {{
                const ability = calculateShootingAbility(shot);
                const isGoal = parseInt(shot.is_goal) === 1;
                const time = parseFloat(shot.time) || 0;
                
                html += `
                    <div class="d-flex justify-content-between align-items-center border-bottom py-2">
                        <div>
                            <strong>${{time.toFixed(0)}}'</strong>
                            ${{isGoal ? 
                                '<span class="badge bg-success">⚽ GOAL</span>' : 
                                '<span class="badge bg-secondary">No Goal</span>'
                            }}
                        </div>
                        <div class="text-end">
                            <div><strong>${{ability.toFixed(1)}}/100</strong></div>
                            <small class="text-muted">vs ${{shot.opponent || 'Unknown'}}</small>
                        </div>
                    </div>
                `;
            }});
            
            shotHistory.innerHTML = html;
        }}

        function showHome() {{
            document.getElementById('homePage').style.display = 'block';
            document.getElementById('playerProfile').style.display = 'none';
            document.getElementById('playerSearch').value = '';
        }}

        // Search functionality
        document.getElementById('playerSearch').addEventListener('input', function() {{
            const query = this.value.toLowerCase().trim();
            const searchResults = document.getElementById('searchResults');
            
            if (query.length < 2) {{
                searchResults.style.display = 'none';
                return;
            }}
            
            const matches = playersData.filter(player => 
                player.Player.toLowerCase().includes(query)
            ).slice(0, 10);
            
            if (matches.length > 0) {{
                let html = '';
                matches.forEach(player => {{
                    html += `
                        <a href="#" class="list-group-item list-group-item-action" 
                           onclick="selectPlayer('${{player.Player}}')">
                            ${{player.Player}} <span class="badge bg-primary float-end">${{player.Overall_Shooting_Rating}}</span>
                        </a>
                    `;
                }});
                searchResults.innerHTML = html;
                searchResults.style.display = 'block';
            }} else {{
                searchResults.style.display = 'none';
            }}
        }});

        function selectPlayer(playerName) {{
            document.getElementById('playerSearch').value = playerName;
            document.getElementById('searchResults').style.display = 'none';
        }}

        // Hide search results when clicking outside
        document.addEventListener('click', function(e) {{
            const searchInput = document.getElementById('playerSearch');
            const searchResults = document.getElementById('searchResults');
            if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {{
                searchResults.style.display = 'none';
            }}
        }});

        // Allow Enter key to search
        document.getElementById('playerSearch').addEventListener('keypress', function(e) {{
            if (e.key === 'Enter') {{
                searchPlayer();
            }}
        }});
    </script>
</body>
</html>'''

# Save the embedded HTML file
with open('player_analysis_embedded.html', 'w', encoding='utf-8') as f:
    f.write(html_content)
