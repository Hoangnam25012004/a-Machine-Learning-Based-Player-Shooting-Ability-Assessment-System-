# A Machine Learning-Based Player Shooting Ability Assessment System

A comprehensive machine learning system for evaluating football player shooting abilities using event-level match data and interactive web visualization.

## 📊 Project Overview

This project develops a multi-dimensional scoring system to assess football player shooting abilities beyond traditional goal-based metrics. Using machine learning techniques on over 229,000 shot attempts from European football leagues, the system provides detailed player rankings and performance insights.

## 🎯 Key Features

- **Machine Learning Model**: Random Forest regression with 84.7% variance explanation (R² = 0.847)
- **Comprehensive Scoring**: Multi-factor evaluation considering shot placement, location, context, and technique
- **Player Rankings**: Analysis of 4,113 players with detailed performance metrics
- **Interactive Web Interface**: Real-time player search and visualization
- **Time-Series Analysis**: Shooting ability progression tracking over time
- **Statistical Validation**: Correlation analysis with actual goal outcomes

## 📁 Project Structure
football-shooting-analysis/

├── train_model.py # Main analysis and model training script

├── create_embedded_website.py # Interactive web application generator

├── player_shooting_ratings.csv # Generated player ratings dataset

├── Player_Shooting_Analysis_Report.html # Academic research report

└── README.md # Project documentation


## 🚀 Getting Started

### Prerequisites
bash
pip install pandas numpy scikit-learn matplotlib


### Dataset

Download the Football Events dataset from Kaggle:
- **Source**: [Football Events Dataset](https://www.kaggle.com/datasets/secareanualin/football-events)
- **File needed**: `events.csv`
- **Coverage**: 900,000+ events from 9,074 games (2011-2017)
- **Leagues**: England, Spain, Germany, Italy, France

### Running the Analysis

1. **Train the Model and Generate Ratings**:
bash
python train_model.py
This will:
- Load and preprocess the dataset
- Train the Random Forest model
- Generate player shooting ability scores
- Create `player_shooting_ratings.csv`
- Display statistical analysis and visualizations

2. **Create Interactive Web Interface**:
bash
python create_embedded_website.py

This generates an HTML file with interactive player analysis features.

## 🧮 Methodology

### Shooting Ability Scoring (0-100 scale)

The system evaluates shots based on six key factors:

1. **Shot Placement (40 points)**: Goal corners, center, post hits, blocks, misses
2. **Shot Outcome (25 points)**: On target, off target, blocked, hit bar
3. **Field Location (20 points)**: Box position, penalty area, long range
4. **Body Part (8 points)**: Right foot, left foot, head technique
5. **Assist Method (4 points)**: Through ball, pass, cross, individual
6. **Situation (3 points)**: Open play, set piece, corner, free kick

### Machine Learning Model

- **Algorithm**: Random Forest Regressor
- **Features**: 6 categorical variables
- **Performance**: R² = 0.847, RMSE = 12.5
- **Validation**: 80/20 train-test split
- **Correlation with goals**: 0.440 (p < 0.001)

### Overall Player Rating

Player ratings combine:
- Average shooting ability score
- Shot consistency (lower standard deviation = bonus)
- Goal conversion rate
- Shot volume factor (minimum 3 shots required)

## 📈 Key Results

### Model Performance
- **Players Analyzed**: 4,113 (minimum 3 shots)
- **Average Rating**: 64.2 ± 12.8
- **Elite Players (≥90)**: 89 players (2.2%)
- **High Quality (≥80)**: 312 players (7.6%)
- **Above Average (≥70)**: 847 players (20.6%)

### Feature Importance
1. Shot Placement: 34.2%
2. Shot Outcome: 28.7%
3. Field Location: 19.8%
4. Body Part: 8.9%
5. Assist Method: 4.7%
6. Situation: 3.7%

## 🌐 Web Interface Features

The interactive web application provides:

- **Player Search**: Autocomplete search functionality
- **Top Players List**: Rankings with shooting ratings
- **Player Profiles**: Detailed statistics and performance metrics
- **Time-Series Charts**: Shooting ability progression over time
- **Shot History**: Individual shot analysis with context

**Web Application Main Interface**

![Web Application Main Interface](https://github.com/Hoangnam25012004/a-Machine-Learning-Based-Player-Shooting-Ability-Assessment-System-/blob/main/images/Picture1.png)

**Player Analysis Dashboard**

![Player Analysis Dashboard](https://github.com/Hoangnam25012004/a-Machine-Learning-Based-Player-Shooting-Ability-Assessment-System-/blob/main/images/Picture2.png)

**Shooting Ability by Shot Sequence Visualization**

![Shooting Ability by Shot Sequence Visualization](https://github.com/Hoangnam25012004/a-Machine-Learning-Based-Player-Shooting-Ability-Assessment-System-/blob/main/images/Picture3.png)

**Statistical Analysis Results**

![Statistical Analysis Results](https://github.com/Hoangnam25012004/a-Machine-Learning-Based-Player-Shooting-Ability-Assessment-System-/blob/main/images/Picture4.png)


## 📊 Sample Output

### Top 5 Players by Overall Shooting Rating:
1. **guidetti** - 94.5 (20 shots, 45% conversion)
2. **daniel guiza** - 93.8 (15 shots, 40% conversion)
3. **grafite** - 92.1 (18 shots, 38.9% conversion)
4. **mario gomez** - 91.7 (25 shots, 44% conversion)
5. **carlos vela** - 90.9 (22 shots, 36.4% conversion)

## 🔬 Research Applications

### For Scouts
- Identify undervalued players with high shooting ability
- Compare players across different leagues and contexts
- Assess shooting consistency and reliability

### For Analysts
- Track player development over time
- Analyze shooting patterns and tendencies
- Evaluate tactical effectiveness in different situations

### For Coaches
- Identify training focus areas for shooting improvement
- Understand optimal shooting situations and contexts
- Monitor player performance trends

## 📝 Academic Publication

The project includes a comprehensive research report (`Player_Shooting_Analysis_Report.html`) following IMRaD structure with:
- Literature review and methodology
- Statistical validation and results
- Discussion of implications and limitations
- IEEE citation format for academic standards

## 🔮 Future Enhancements

- **Defensive Pressure**: Incorporate tracking data for defensive context
- **Goalkeeper Analysis**: Account for goalkeeper quality in shot evaluation
- **Temporal Modeling**: Advanced time-series analysis of player development
- **xG Comparison**: Validation against established Expected Goals models
- **Multi-Skill Analysis**: Extension to passing, defending, and other skills

## 📄 License

This project is for educational and research purposes. Dataset usage follows Kaggle's terms of service.

## 🙏 Acknowledgments

- **Dataset**: Secareanualin (Kaggle) for the Football Events dataset
- **Data Sources**: BBC, ESPN, OneFootball for original match commentary
- **Community**: Football analytics community for inspiration and validation

## 📧 Contact

**Author**: Đặng Hoàng Nam  
**Email**: dnam2501@gmail.com  
**Date**: June 2, 2025

---

*This project demonstrates the application of machine learning in sports analytics, providing quantitative insights into football player performance beyond traditional statistical measures.*
