"""
Simple example of how to use the Style Recommender after training.
"""

from predict_style import StyleRecommender

# Initialize the recommender (make sure you've trained the model first)
recommender = StyleRecommender()

# Get user input
print("="*60)
print("Martial Arts Style Recommender")
print("="*60)
print("\nPlease provide the following information:\n")

experience = input("Experience level (Beginner/Intermediate/Advanced): ").strip()
time_per_week = input("Time per week (1-2 hours/3-5 hours/6-10 hours/10+ hours): ").strip()
budget_range = input("Budget range (<1000/1000-3000/3000-6000/6000+): ").strip()
format_type = input("Format (Online/Offline/Hybrid): ").strip()
location = input("Location (Mumbai/Delhi/Chennai/Bangalore): ").strip()

print("\nAvailable styles: Krav Maga, BJJ, Kung Fu, Taekwondo, Boxing, Wrestling, Muay Thai, MMA, Judo, Karate")
styles_input = input("Styles you're interested in (comma-separated, or press Enter for none): ").strip()
styles_interested = [s.strip() for s in styles_input.split(',')] if styles_input else None

print("\nAvailable goals: Weight Loss, Mobility, Flexibility, Tournaments, Competition, Fitness, Building Confidence, Learning Traditional Arts, Stress Relief, Self-defense, Mental Discipline")
goals_input = input("Your goals (comma-separated, or press Enter for none): ").strip()
goals = [g.strip() for g in goals_input.split(',')] if goals_input else None

# Get recommendation
print("\n" + "="*60)
print("Analyzing your preferences...")
print("="*60)

result = recommender.predict(
    experience=experience,
    time_per_week=time_per_week,
    budget_range=budget_range,
    format_type=format_type,
    location=location,
    styles_interested=styles_interested,
    goals=goals
)

# Display results
print(f"\n🎯 Recommended Style: {result['recommended_style']}")
print(f"📊 Confidence: {result['confidence']:.1%}")
print("\n🏆 Top 3 Recommendations:")
for i, rec in enumerate(result['top_3_recommendations'], 1):
    print(f"   {i}. {rec['style']}: {rec['probability']:.1%}")

print("\n" + "="*60)

