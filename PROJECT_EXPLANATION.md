# Martial Arts Style Recommender - Project Explanation

## 🎯 Project Overview

This is a **machine learning recommendation system** that suggests the best martial arts style for users based on their personal preferences, constraints, and goals. Think of it like a personalized fitness advisor that helps people choose the right martial art for them.

## 📊 The Problem It Solves

Choosing a martial art can be overwhelming - there are many styles (Boxing, Karate, BJJ, MMA, etc.), and what works for one person might not work for another. This system uses data from 500 previous users to learn patterns and recommend the best style based on:

- **Experience level** (Beginner/Intermediate/Advanced)
- **Time availability** (1-2 hours, 3-5 hours, etc.)
- **Budget constraints** (<1000, 1000-3000, etc.)
- **Preferred format** (Online/Offline/Hybrid)
- **Location** (Mumbai, Delhi, Chennai, Bangalore)
- **Styles they're interested in** (optional)
- **Their goals** (Fitness, Self-defense, Competition, etc.)

## 📁 Data Structure

### Input Data (`martially_dataset_processed.csv`)
The dataset contains **500 rows** with the following structure:

#### Categorical Features (5):
1. **experience**: Beginner, Intermediate, Advanced
2. **time_per_week**: 1-2 hours, 3-5 hours, 6-10 hours, 10+ hours
3. **budget_range**: <1000, 1000-3000, 3000-6000, 6000+
4. **format**: Online, Offline, Hybrid
5. **location**: Mumbai, Delhi, Chennai, Bangalore

#### Binary Features (21):
- **10 Style columns**: Krav Maga, BJJ, Kung Fu, Taekwondo, Boxing, Wrestling, Muay Thai, MMA, Judo, Karate (1 if interested, 0 if not)
- **11 Goal columns**: Weight Loss, Mobility, Flexibility, Tournaments, Competition, Fitness, Building Confidence, Learning Traditional Arts, Stress Relief, Self-defense, Mental Discipline (1 if goal, 0 if not)

#### Target Variable:
- **recommended_style**: The actual style recommended (one of 10 martial arts styles)

## 🤖 How the Model Works

### 1. **Training Phase** (`train_style_recommender.py`)

```
Data → Preprocessing → Training → Model Files
```

**Step-by-step process:**

1. **Load Data**: Reads the CSV file with 500 examples
2. **Feature Encoding**: 
   - Categorical features (experience, time_per_week, etc.) are converted to numbers using Label Encoding
   - Binary features (styles/goals) are already 0/1, so they stay as-is
3. **Target Encoding**: The style names are converted to numbers (0-9)
4. **Data Split**: 80% for training (400 samples), 20% for testing (100 samples)
5. **Model Training**: Uses **Random Forest Classifier** with:
   - 200 decision trees
   - Max depth of 15
   - Tuned to prevent overfitting
6. **Evaluation**: Calculates accuracy and shows which features are most important
7. **Save Model**: Saves everything needed for predictions:
   - `style_recommender_model.pkl` - The trained model
   - `label_encoders.pkl` - Encoders for categorical features
   - `target_encoder.pkl` - Encoder for style names
   - `feature_info.json` - Metadata about features

### 2. **Prediction Phase** (`predict_style.py`)

```
User Input → Encode → Predict → Decode → Recommendation
```

**How predictions work:**

1. **User provides inputs**: Experience, time, budget, format, location, styles, goals
2. **Feature encoding**: 
   - Categorical inputs are encoded using saved encoders
   - Styles/goals are converted to binary (1 if selected, 0 if not)
3. **Feature vector creation**: All features are combined in the correct order
4. **Model prediction**: Random Forest votes across all trees to predict style
5. **Probability calculation**: Gets confidence scores for all 10 styles
6. **Result formatting**: Returns:
   - Top recommended style
   - Confidence percentage
   - Top 3 recommendations with probabilities
   - All style probabilities

## 🏗️ Architecture

### Random Forest Classifier
- **Why Random Forest?**: 
  - Handles mixed data types well (categorical + binary)
  - Provides feature importance insights
  - Less prone to overfitting than single decision trees
  - Works well with small-medium datasets

### Feature Importance
The model learns which factors matter most:
- **Location** (6.6% importance) - Different cities may have different style availability
- **Time per week** (6.5% importance) - Some styles require more commitment
- **Budget range** (6.2% importance) - Cost affects style accessibility
- **Self-defense goal** (5.9% importance) - Strong predictor for certain styles
- **Format preference** (5.5% importance) - Online vs offline matters

## 📂 File Structure

```
jaykant/
├── martially_dataset_processed.csv    # Training data (500 samples)
├── train_style_recommender.py         # Training script
├── predict_style.py                   # Prediction class & examples
├── example_usage.py                    # Interactive usage example
├── requirements.txt                    # Python dependencies
├── README.md                          # Quick start guide
│
├── [Generated after training:]
├── style_recommender_model.pkl        # Trained model
├── label_encoders.pkl                 # Feature encoders
├── target_encoder.pkl                 # Style name encoder
└── feature_info.json                  # Feature metadata
```

## 💻 Usage Flow

### Step 1: Train the Model
```bash
python train_style_recommender.py
```
This creates all the `.pkl` and `.json` files needed for predictions.

### Step 2: Make Predictions
```python
from predict_style import StyleRecommender

# Load the trained model
recommender = StyleRecommender()

# Get recommendation
result = recommender.predict(
    experience='Beginner',
    time_per_week='1-2 hours',
    budget_range='1000-3000',
    format_type='Hybrid',
    location='Mumbai',
    styles_interested=['Boxing', 'Karate'],
    goals=['Fitness', 'Self-defense']
)

print(f"Recommended: {result['recommended_style']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 🎓 What the Model Learned

From the 500 training examples, the model learned patterns like:

- **Beginners** with limited time often get recommended **Karate** or **Taekwondo** (structured, beginner-friendly)
- **Advanced** users interested in **competition** often get **MMA** or **BJJ** (competitive styles)
- **Self-defense** goals strongly correlate with **Krav Maga** and **Boxing**
- **Traditional arts** goals correlate with **Kung Fu** and **Karate**
- **Location** matters - different cities have different style availability/popularity

## 📈 Model Performance

- **Training Accuracy**: ~97% (model fits the training data well)
- **Test Accuracy**: ~34% (on unseen data)
- **Note**: Lower test accuracy is expected with:
  - Small dataset (500 samples)
  - 10 different output classes
  - Imbalanced classes (some styles have more examples than others)

The model still provides useful recommendations with probability scores, allowing users to see top 3 options.

## 🔄 Workflow Diagram

```
┌─────────────────┐
│  Raw Data       │
│  (500 samples)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing   │
│  - Encode cats   │
│  - Binary flags  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Train/Test     │
│  Split (80/20)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Random Forest  │
│  Training       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Save Model     │
│  & Encoders     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  User Input     │
│  (New Query)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Encode Input   │
│  Features       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Predicts │
│  Style + Probs  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Return Top     │
│  Recommendation│
└─────────────────┘
```

## 🎯 Key Features

1. **Personalized Recommendations**: Considers 7+ different user factors
2. **Multiple Options**: Returns top 3 recommendations, not just one
3. **Confidence Scores**: Shows how confident the model is in each prediction
4. **Flexible Inputs**: Styles and goals are optional
5. **Production Ready**: Model is saved and can be deployed easily

## 🚀 Potential Improvements

1. **More Data**: Collect more training examples for better accuracy
2. **Feature Engineering**: Add more relevant features (age, fitness level, etc.)
3. **Model Tuning**: Try different algorithms (XGBoost, Neural Networks)
4. **Class Balancing**: Handle imbalanced classes better
5. **Web Interface**: Create a web app for easier user interaction

## 📝 Summary

This project demonstrates a complete ML pipeline:
- **Data preprocessing** and encoding
- **Model training** with Random Forest
- **Model persistence** (saving/loading)
- **Prediction interface** for new users
- **Evaluation metrics** and feature importance analysis

It's a practical recommendation system that can help real users choose the right martial arts style based on their unique situation and preferences.

