import pandas as pd
import numpy as np
import joblib
import json

class StyleRecommender:
    def __init__(self, model_path='style_recommender_model.pkl', 
                 encoders_path='label_encoders.pkl',
                 target_encoder_path='target_encoder.pkl',
                 feature_info_path='feature_info.json'):
        """Initialize the Style Recommender with trained model and encoders."""
        self.model = joblib.load(model_path)
        self.label_encoders = joblib.load(encoders_path)
        self.target_encoder = joblib.load(target_encoder_path)
        
        with open(feature_info_path, 'r') as f:
            self.feature_info = json.load(f)
        
        # Available styles and goals
        self.available_styles = ['Krav Maga', 'BJJ', 'Kung Fu', 'Taekwondo', 
                                'Boxing', 'Wrestling', 'Muay Thai', 'MMA', 
                                'Judo', 'Karate']
        self.available_goals = ['Weight Loss', 'Mobility', 'Flexibility', 
                               'Tournaments', 'Competition', 'Fitness',
                               'Building Confidence', 'Learning Traditional Arts',
                               'Stress Relief', 'Self-defense', 'Mental Discipline']
    
    def predict(self, experience, time_per_week, budget_range, format_type, 
                location, styles_interested=None, goals=None):
        """
        Predict recommended style based on user inputs.
        
        Parameters:
        -----------
        experience : str
            One of: 'Beginner', 'Intermediate', 'Advanced'
        time_per_week : str
            One of: '1-2 hours', '3-5 hours', '6-10 hours', '10+ hours'
        budget_range : str
            One of: '<1000', '1000-3000', '3000-6000', '6000+'
        format_type : str
            One of: 'Online', 'Offline', 'Hybrid'
        location : str
            One of: 'Mumbai', 'Delhi', 'Chennai', 'Bangalore'
        styles_interested : list, optional
            List of styles the user is interested in (e.g., ['Boxing', 'BJJ'])
        goals : list, optional
            List of goals (e.g., ['Fitness', 'Self-defense'])
        
        Returns:
        --------
        dict : Prediction results with recommended style and probabilities
        """
        # Create feature vector
        features = {}
        
        # Encode categorical features
        categorical_cols = self.feature_info['categorical_cols']
        features['experience'] = self.label_encoders['experience'].transform([experience])[0]
        features['time_per_week'] = self.label_encoders['time_per_week'].transform([time_per_week])[0]
        features['budget_range'] = self.label_encoders['budget_range'].transform([budget_range])[0]
        features['format'] = self.label_encoders['format'].transform([format_type])[0]
        features['location'] = self.label_encoders['location'].transform([location])[0]
        
        # Initialize binary features (styles and goals) to 0
        for style in self.available_styles:
            features[style] = 0
        for goal in self.available_goals:
            features[f'goal_{goal}'] = 0
        
        # Set styles interested
        if styles_interested:
            for style in styles_interested:
                if style in self.available_styles:
                    features[style] = 1
        
        # Set goals
        if goals:
            for goal in goals:
                if goal in self.available_goals:
                    features[f'goal_{goal}'] = 1
        
        # Create DataFrame with correct column order
        feature_df = pd.DataFrame([features])
        feature_df = feature_df[self.feature_info['all_features']]
        
        # Make prediction
        prediction_encoded = self.model.predict(feature_df)[0]
        prediction_proba = self.model.predict_proba(feature_df)[0]
        
        # Decode prediction
        recommended_style = self.target_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get probabilities for all styles
        style_probs = {}
        for i, style in enumerate(self.target_encoder.classes_):
            style_probs[style] = float(prediction_proba[i])
        
        # Sort by probability
        sorted_probs = sorted(style_probs.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'recommended_style': recommended_style,
            'confidence': float(style_probs[recommended_style]),
            'all_probabilities': dict(sorted_probs),
            'top_3_recommendations': [{'style': style, 'probability': prob} 
                                     for style, prob in sorted_probs[:3]]
        }


def main():
    """Example usage of the Style Recommender."""
    print("Loading Style Recommender model...")
    recommender = StyleRecommender()
    
    print("\n" + "="*60)
    print("Style Recommender - Example Usage")
    print("="*60)
    
    # Example 1: Beginner user
    print("\nExample 1: Beginner user")
    print("-" * 60)
    result = recommender.predict(
        experience='Beginner',
        time_per_week='1-2 hours',
        budget_range='1000-3000',
        format_type='Hybrid',
        location='Mumbai',
        styles_interested=['Boxing', 'Karate'],
        goals=['Fitness', 'Self-defense']
    )
    print(f"Recommended Style: {result['recommended_style']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nTop 3 Recommendations:")
    for rec in result['top_3_recommendations']:
        print(f"  - {rec['style']}: {rec['probability']:.2%}")
    
    # Example 2: Advanced user
    print("\n\nExample 2: Advanced user")
    print("-" * 60)
    result = recommender.predict(
        experience='Advanced',
        time_per_week='10+ hours',
        budget_range='3000-6000',
        format_type='Offline',
        location='Delhi',
        styles_interested=['MMA', 'BJJ'],
        goals=['Competition', 'Tournaments']
    )
    print(f"Recommended Style: {result['recommended_style']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nTop 3 Recommendations:")
    for rec in result['top_3_recommendations']:
        print(f"  - {rec['style']}: {rec['probability']:.2%}")
    
    # Example 3: Intermediate user
    print("\n\nExample 3: Intermediate user")
    print("-" * 60)
    result = recommender.predict(
        experience='Intermediate',
        time_per_week='3-5 hours',
        budget_range='<1000',
        format_type='Online',
        location='Bangalore',
        styles_interested=['Muay Thai'],
        goals=['Stress Relief', 'Fitness']
    )
    print(f"Recommended Style: {result['recommended_style']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nTop 3 Recommendations:")
    for rec in result['top_3_recommendations']:
        print(f"  - {rec['style']}: {rec['probability']:.2%}")


if __name__ == '__main__':
    main()

