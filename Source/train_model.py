import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv(r"D:\Central forward stat\events.csv")

# Display first 5 rows
print(df.head())

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def create_shooting_ability_target(row):
    """
    Create target shooting ability score (0-100) based on shot quality factors
    """
    if row['event_type'] != 1:  # Only for shot attempts
        return np.nan
    
    score = 0
    
    # Shot placement scoring (40 points max)
    shot_place_scores = {
        3: 40,   # Bottom left corner - excellent
        4: 40,   # Bottom right corner - excellent  
        5: 35,   # Centre of the goal - very good
        11: 38,  # Top centre of the goal - very good
        12: 40,  # Top left corner - excellent
        13: 40,  # Top right corner - excellent
        7: 25,   # Hits the bar - good attempt
        1: 15,   # Bit too high - poor
        2: 10,   # Blocked - poor
        6: 5,    # High and wide - very poor
        8: 8,    # Misses to the left - very poor
        9: 8,    # Misses to the right - very poor
        10: 5    # Too high - very poor
    }
    score += shot_place_scores.get(row['shot_place'], 0)
    
    # Shot outcome scoring (25 points max)
    outcome_scores = {
        1: 25,   # On target - excellent
        4: 20,   # Hit the bar - good
        2: 5,    # Off target - poor
        3: 10    # Blocked - below average
    }
    score += outcome_scores.get(row['shot_outcome'], 0)
    
    # Location scoring (20 points max)
    location_scores = {
        3: 20,   # Centre of the box - excellent position
        13: 20,  # Very close range - excellent
        14: 18,  # Penalty spot - very good
        9: 15,   # Left side of the box - good
        11: 15,  # Right side of the box - good
        10: 15,  # Left side of six yard box - good
        12: 15,  # Right side of six yard box - good
        15: 10,  # Outside the box - average
        16: 8,   # Long range - below average
        6: 5,    # Difficult angle and long range - poor
        7: 6,    # Difficult angle on the left - poor
        8: 6,    # Difficult angle on the right - poor
        17: 3,   # More than 35 yards - very poor
        18: 2,   # More than 40 yards - very poor
        1: 8,    # Attacking half - average
        4: 7,    # Left wing - below average
        5: 7     # Right wing - below average
    }
    score += location_scores.get(row['location'], 0)
    
    # Body part scoring (8 points max)
    bodypart_scores = {
        1: 8,    # Right foot - good technique
        2: 8,    # Left foot - good technique  
        3: 6     # Head - slightly lower technique
    }
    score += bodypart_scores.get(row['bodypart'], 0)
    
    # Assist method bonus (4 points max)
    assist_scores = {
        0: 2,    # None - individual skill
        1: 4,    # Pass - good setup
        2: 3,    # Cross - decent setup
        3: 3,    # Headed pass - decent setup
        4: 4     # Through ball - excellent setup
    }
    score += assist_scores.get(row['assist_method'], 0)
    
    # Situation bonus (3 points max)
    situation_scores = {
        1: 3,    # Open play - good
        2: 2,    # Set piece - average
        3: 2,    # Corner - average
        4: 2     # Free kick - average
    }
    score += situation_scores.get(row['situation'], 0)
    
    return min(score, 100)  # Cap at 100

# Create shooting ability scores for all shot attempts
df['shooting_ability_score'] = df.apply(create_shooting_ability_target, axis=1)

# Filter for shot attempts only
shot_data = df[df['event_type'] == 1].copy()
shot_data = shot_data.dropna(subset=['shooting_ability_score'])

print(f"\nDataset Info:")
print(f"Total shot attempts: {len(shot_data)}")
print(f"Features available: shot_place, shot_outcome, location, bodypart, assist_method, situation")

# Prepare features for training
features = ['shot_place', 'shot_outcome', 'location', 'bodypart', 'assist_method', 'situation']
X = shot_data[features]
y = shot_data['shooting_ability_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.3f}")
print(f"Root Mean Squared Error: {np.sqrt(mse):.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance:")
print(feature_importance)

# Show some predictions vs actual
comparison = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred[:10],
    'Difference': abs(y_test.values[:10] - y_pred[:10])
})
print(f"\nSample Predictions vs Actual (first 10):")
print(comparison)

# Function to predict shooting ability for new shots
def predict_shooting_ability(shot_place, shot_outcome, location, bodypart, assist_method, situation):
    """
    Predict shooting ability score for a new shot
    """
    features_array = np.array([[shot_place, shot_outcome, location, bodypart, assist_method, situation]])
    prediction = rf_model.predict(features_array)[0]
    return min(max(prediction, 0), 100)  # Ensure score is between 0-100

# Example predictions
print(f"\nExample Predictions:")
print(f"Perfect shot (bottom corner, on target, center box, right foot, through ball, open play): {predict_shooting_ability(3, 1, 3, 1, 4, 1):.1f}")
print(f"Poor shot (high and wide, off target, long range, head, no assist, set piece): {predict_shooting_ability(6, 2, 16, 3, 0, 2):.1f}")
print(f"Average shot (center goal, on target, outside box, right foot, pass, open play): {predict_shooting_ability(5, 1, 15, 1, 1, 1):.1f}")

# Add analysis of the shooting ability distribution
print(f"\nShooting Ability Score Distribution:")
print(f"Mean: {shot_data['shooting_ability_score'].mean():.2f}")
print(f"Median: {shot_data['shooting_ability_score'].median():.2f}")
print(f"Standard Deviation: {shot_data['shooting_ability_score'].std():.2f}")
print(f"Min: {shot_data['shooting_ability_score'].min():.2f}")
print(f"Max: {shot_data['shooting_ability_score'].max():.2f}")

# Show distribution by score ranges
score_ranges = pd.cut(shot_data['shooting_ability_score'], 
                     bins=[0, 20, 40, 60, 80, 100], 
                     labels=['Poor (0-20)', 'Below Average (21-40)', 'Average (41-60)', 'Good (61-80)', 'Excellent (81-100)'])
print(f"\nShots by Quality Level:")
print(score_ranges.value_counts().sort_index())

# Analyze by player performance
player_stats = shot_data.groupby('player').agg({
    'shooting_ability_score': ['count', 'mean', 'std'],
    'is_goal': 'sum'
}).round(2)

player_stats.columns = ['Total_Shots', 'Avg_Shooting_Ability', 'Std_Shooting_Ability', 'Goals']
player_stats = player_stats[player_stats['Total_Shots'] >= 5]  # Players with at least 5 shots
player_stats['Goal_Rate'] = (player_stats['Goals'] / player_stats['Total_Shots'] * 100).round(1)
player_stats = player_stats.sort_values('Avg_Shooting_Ability', ascending=False)

print(f"\nTop 10 Players by Average Shooting Ability (min 5 shots):")
print(player_stats.head(10))

# Correlation between shooting ability and actual goals
correlation = shot_data['shooting_ability_score'].corr(shot_data['is_goal'])
print(f"\nCorrelation between Shooting Ability Score and Goals: {correlation:.3f}")

# Add visualization if matplotlib is available
try:
    import matplotlib.pyplot as plt
    
    # Create shooting ability distribution plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(shot_data['shooting_ability_score'], bins=20, alpha=0.7, color='skyblue')
    plt.title('Distribution of Shooting Ability Scores')
    plt.xlabel('Shooting Ability Score')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 2)
    plt.scatter(shot_data['shooting_ability_score'], shot_data['is_goal'], alpha=0.5)
    plt.title('Shooting Ability vs Goals')
    plt.xlabel('Shooting Ability Score')
    plt.ylabel('Goal (1) or No Goal (0)')
    
    plt.subplot(2, 2, 3)
    feature_importance.plot(x='feature', y='importance', kind='bar')
    plt.title('Feature Importance in Shooting Ability Model')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Predicted vs Actual Shooting Ability')
    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')
    
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("Matplotlib not available for visualization")

# Create overall player shooting ratings based on historical performance
def calculate_overall_player_rating(player_data):
    """
    Calculate overall shooting rating for a player based on their shot history
    """
    if len(player_data) == 0:
        return 0
    
    # Base rating from average shooting ability score
    avg_ability = player_data['shooting_ability_score'].mean()
    
    # Consistency factor (lower std deviation = more consistent = bonus)
    std_ability = player_data['shooting_ability_score'].std()
    consistency_bonus = max(0, (10 - std_ability/2))  # Up to 10 points for consistency
    
    # Volume factor (more shots = more reliable rating)
    shot_count = len(player_data)
    volume_factor = min(1.0, shot_count / 20)  # Full weight at 20+ shots
    
    # Goal conversion bonus
    goal_rate = player_data['is_goal'].mean()
    conversion_bonus = goal_rate * 15  # Up to 15 points for high conversion
    
    # Calculate final rating
    overall_rating = (avg_ability + consistency_bonus + conversion_bonus) * volume_factor
    
    return min(overall_rating, 100)  # Cap at 100

# Calculate overall ratings for all players
player_overall_ratings = []

for player in shot_data['player'].unique():
    if pd.isna(player):  # Skip NaN players
        continue
        
    player_shots = shot_data[shot_data['player'] == player]
    
    if len(player_shots) >= 3:  # Minimum 3 shots for rating
        overall_rating = calculate_overall_player_rating(player_shots)
        
        player_overall_ratings.append({
            'Player': player,
            'Overall_Shooting_Rating': round(overall_rating, 1),
            'Total_Shots': len(player_shots),
            'Goals': player_shots['is_goal'].sum(),
            'Goal_Rate_%': round(player_shots['is_goal'].mean() * 100, 1),
            'Avg_Shot_Quality': round(player_shots['shooting_ability_score'].mean(), 1),
            'Shot_Consistency': round(player_shots['shooting_ability_score'].std(), 1),
            'Best_Shot': round(player_shots['shooting_ability_score'].max(), 1),
            'Worst_Shot': round(player_shots['shooting_ability_score'].min(), 1)
        })

# Create DataFrame and sort by overall rating
overall_ratings_df = pd.DataFrame(player_overall_ratings)
overall_ratings_df = overall_ratings_df.sort_values('Overall_Shooting_Rating', ascending=False)

print(f"\n" + "="*80)
print(f"OVERALL PLAYER SHOOTING RATINGS (Sorted Highest to Lowest)")
print(f"="*80)
print(f"Rating considers: Shot Quality + Consistency + Goal Conversion + Shot Volume")
print(f"Minimum 3 shots required for rating")
print(f"\nTop 20 Players:")
print(overall_ratings_df.head(20).to_string(index=False))

print(f"\n" + "="*50)
print(f"RATING BREAKDOWN FOR TOP 5 PLAYERS:")
print(f"="*50)

for i, (_, player) in enumerate(overall_ratings_df.head(5).iterrows()):
    print(f"\n{i+1}. {player['Player']} - Overall Rating: {player['Overall_Shooting_Rating']}")
    print(f"   • Total Shots: {player['Total_Shots']}")
    print(f"   • Goals: {player['Goals']} ({player['Goal_Rate_%']}% conversion)")
    print(f"   • Average Shot Quality: {player['Avg_Shot_Quality']}/100")
    print(f"   • Shot Consistency: {player['Shot_Consistency']} (lower = more consistent)")
    print(f"   • Best Shot: {player['Best_Shot']}/100")
    print(f"   • Worst Shot: {player['Worst_Shot']}/100")

# Save the ratings to CSV file
overall_ratings_df.to_csv('player_shooting_ratings.csv', index=False)
print(f"\n✅ Player shooting ratings saved to 'player_shooting_ratings.csv'")

# Summary statistics
print(f"\n" + "="*50)
print(f"SUMMARY STATISTICS:")
print(f"="*50)
print(f"Total players rated: {len(overall_ratings_df)}")
print(f"Average overall rating: {overall_ratings_df['Overall_Shooting_Rating'].mean():.1f}")
print(f"Highest rating: {overall_ratings_df['Overall_Shooting_Rating'].max():.1f}")
print(f"Lowest rating: {overall_ratings_df['Overall_Shooting_Rating'].min():.1f}")
print(f"Players with rating ≥ 70: {len(overall_ratings_df[overall_ratings_df['Overall_Shooting_Rating'] >= 70])}")
print(f"Players with rating ≥ 80: {len(overall_ratings_df[overall_ratings_df['Overall_Shooting_Rating'] >= 80])}")
print(f"Players with rating ≥ 90: {len(overall_ratings_df[overall_ratings_df['Overall_Shooting_Rating'] >= 90])}")
