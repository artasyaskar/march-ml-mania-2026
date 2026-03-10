# March ML Mania 2026 - NCAA Tournament Prediction System

## Overview

This project implements a robust machine learning pipeline to predict outcomes of NCAA men's and women's basketball tournament games for the 2026 March ML Mania competition. The system combines multiple modeling approaches including Elo ratings, efficiency metrics, and gradient-boosted trees with careful calibration and ensemble blending to produce well-calibrated probability estimates.

## Competition Context

March ML Mania is an annual data science competition focused on predicting NCAA basketball tournament outcomes. Participants must estimate win probabilities for every possible tournament matchup, with evaluation based on Brier score (log loss). The challenge requires handling historical data from multiple seasons, dealing with asymmetric team strength distributions, and producing reliable probability estimates rather than just point predictions.

## Technical Architecture

### Data Pipeline

The system processes extensive historical tournament and regular season data including:
- Game results with detailed box score statistics
- Team seeding and tournament structure
- Massey rating systems for additional strength signals
- Conference affiliations and geographical information

Data is organized by season and gender (men's/women's) with careful handling of feature consistency across years.

### Feature Engineering

#### Core Features
- **Elo Ratings**: Custom implementation with home advantage adjustment and margin-of-victory weighting
- **Seed Differences**: Tournament seeding disparities as strength indicators
- **Play-in Flags**: Binary indicators for teams entering through play-in games

#### Advanced Metrics
- **Efficiency Differentials**: Offensive and defensive efficiency ratings (points per 100 possessions)
- **Pace Metrics**: Tempo (possessions per game) and style indicators
- **Advanced Stats**: Effective field goal percentage, turnover rates, offensive rebounding, free throw rates
- **Recent Form**: Rolling averages for last 10 games, last 30 days, and exponential decay weighted recent performance

### Modeling Approach

#### Multi-Model Ensemble
The system employs three complementary modeling approaches:

1. **Elo+Seed Logistic Regression**: Baseline model using fundamental strength indicators
2. **Efficiency Logistic Regression**: Model focusing on advanced team performance metrics
3. **LightGBM Gradient Boosting**: Non-linear model capturing complex feature interactions

#### Ensemble Strategy
- Out-of-fold predictions during training to prevent leakage
- Isotonic regression calibration for each model
- Logistic regression meta-blender combining all three models
- Careful feature selection and regularization to prevent overfitting

### Cross-Validation Framework

- GroupKFold validation by season to simulate tournament prediction
- Season-wise splitting prevents temporal leakage
- Comprehensive metrics tracking (Brier score, calibration, coverage)
- Early stopping with validation monitoring

## Key Technical Challenges and Solutions

### Women's Tournament Handling

The women's tournament presents unique challenges due to:
- Different data availability and quality compared to men's
- Potential feature distribution shifts
- Historical data limitations

**Solution Implemented:**
- Robust Elo-only fallback with shrinkage factor (0.35) to prevent extreme predictions
- Neutral prior blending (30% weight to 0.5) to avoid degenerate probability blocks
- Softer probability bounds [0.02, 0.98] for women's matchups
- Post-scoring safety checks to ensure no predictions at extreme values

### Probability Calibration

- Isotonic regression for each component model
- Final ensemble calibration through meta-blender
- Explicit clipping to prevent 0/1 probabilities while maintaining realistic ranges
- Separate calibration strategies for men's and women's tournaments

### Feature Robustness

- Missing value handling with domain-aware imputation
- Outlier detection and capping for extreme values
- Feature stability analysis across seasons
- Automatic feature selection based on coverage and signal quality

## Performance and Validation

### Model Selection
- Hyperparameter tuning via cross-validation
- Model comparison using Brier score and calibration plots
- Ensemble weight optimization
- Robustness testing across different tournament scenarios

### Evaluation Metrics
- Primary: Brier score (log loss)
- Secondary: Calibration reliability, coverage analysis
- Season-wise performance tracking
- Gender-specific performance monitoring

## Implementation Details

### Technology Stack
- **Language**: Python 3.11
- **Core Libraries**: pandas, numpy, scikit-learn, lightgbm
- **Data Processing**: Efficient vectorized operations with careful memory management
- **Model Persistence**: joblib for model serialization
- **Configuration**: Dataclass-based configuration management

### Code Organization
- Modular design with clear separation of concerns
- Feature engineering pipelines with reusable components
- Comprehensive error handling and logging
- Extensive documentation and type hints

### Performance Optimizations
- Vectorized feature computation
- Memory-efficient data structures
- Parallel processing where applicable
- Caching of expensive computations

## Usage Instructions

### Training Models
```bash
python -m mmmlm.train --data-root /path/to/data --outdir /path/to/artifacts --start-season 2003 --end-season 2025 --use-detailed
```

### Generating Predictions
```bash
python -m mmmlm.predict --data-root /path/to/data --artifacts /path/to/models.joblib --out submission.csv --season 2026
```

### Key Configuration Options
- Season range for training data
- Feature inclusion (basic vs. detailed statistics)
- Model hyperparameters via configuration classes
- Cross-validation fold structure

## Project Structure

```
src/mmmlm/
├── predict.py          # Main prediction pipeline
├── train.py            # Model training and validation
├── lgbm_model.py       # LightGBM model utilities
├── features.py         # Feature engineering pipeline
├── elo.py             # Elo rating system implementation
└── data.py            # Data loading and preprocessing
```

## Key Innovations

1. **Adaptive Women's Model**: Recognizes and addresses the unique challenges of women's tournament prediction through specialized fallback mechanisms

2. **Robust Ensemble Design**: Combines multiple modeling philosophies while maintaining interpretability and calibration

3. **Temporal Validation**: Season-wise cross-validation that truly reflects the tournament prediction task

4. **Probability Safety**: Multiple layers of protection against degenerate predictions while preserving model expressiveness

## Future Enhancements

- Integration of additional rating systems (e.g., NET, KenPom)
- Advanced feature selection using mutual information
- Bayesian model averaging for uncertainty quantification
- Real-time prediction updates during tournament play
- Extended validation through historical tournament simulations

## Acknowledgments

This project leverages publicly available NCAA tournament data and builds upon established sports analytics methodologies. The implementation respects data usage guidelines and competition requirements while advancing the state-of-the-art in tournament outcome prediction.
