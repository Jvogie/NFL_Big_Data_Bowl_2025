# Pre-snap Pressure Prediction Index (PPPI)

## Overview
This project introduces the Pre-snap Pressure Prediction Index (PPPI), a novel metric for predicting quarterback pressure in the NFL using pre-snap player tracking data. The PPPI model analyzes defensive alignments, player movements, and game situations to estimate the probability of pressure on any given play.

## Features
- Pre-snap defensive alignment analysis
- Player movement and acceleration tracking
- Offensive line formation evaluation
- Situational pressure assessment
- Advanced feature engineering
- Multiple model comparison (XGBoost, CatBoost, etc.)
- Visualization tools for play analysis

## Project Structure
```
NFL_Big_Data_Bowl_2025/
├── pppi/
│   ├── __init__.py
│   ├── evaluation.py      # Model evaluation and PPPI calculation
│   ├── feature_engineering.py  # Feature creation and processing
│   ├── models/           # Model implementations
│   └── visualization.py  # Visualization functions
├── main.py              # Main execution script
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/NFL_Big_Data_Bowl_2025.git
cd NFL_Big_Data_Bowl_2025
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Prepare your data:
   - Ensure you have the NFL tracking data in the correct format
   - Data should include pre-snap frames and player tracking information
   - The data I used can be found [here](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025)

2. Run the main script:
```bash
python main.py
```

3. View the results:
   - PPPI scores will be calculated for each play
   - Visualizations will be generated in the output directory
   - Model performance metrics will be displayed

## Features Description
The model uses various features organized into categories:

### Defensive Position Features
- Basic position metrics (distances, box presence)
- Aggregate position statistics

### Movement Features
- Speed and acceleration metrics
- Direction and orientation analysis

### Offensive Line Features
- Line structure measurements
- Formation analysis

### Matchup Features
- Personnel ratios and matchups

### Feature Interactions
- Complex metrics combining multiple factors
- Situational pressure indicators

For a complete list of features and their definitions, see the feature engineering documentation.

## Model Performance
The project implements and compares multiple machine learning models:
- XGBoost
- CatBoost
- Random Forest
- Neural Networks
- LightGBM

Model selection is based on ROC-AUC scores and cross-validation performance.

## Visualization
The project includes visualization tools for:
- Pre-snap alignments
- PPPI distribution
- Feature importance
- ROC curves
- Model comparison

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- NFL Big Data Bowl 2025
- NFL Next Gen Stats
- All contributors and participants

## Contact
For questions or feedback, please open an issue on GitHub. 
