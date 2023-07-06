
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

class Emotion:
    def __init__(self, gpt, emotions):
        self.emotions = emotions
        self.sia = SentimentIntensityAnalyzer()
        self.gpt = gpt

    def reset(self):
        self.emotions = {
            "Openness: 78", "Conscientousness: 42", "Extroversion: 80", "Agreeableness: 53", "Neuroticism: 42", "Joy-like: 227", "Trust-like: 212", "Fear-like: 111", "Surprise-like: 83", "Sadness-like: 0", "Disgust-like: 0", "Anger-like: 0", "Anticipation-like: 97"
        }
        return self.emotions

    def analyze_emotion(self, thought, emotions):
        system_prompt = """I am an artifical cognitive entity. I am capable of feeling emotions.
        I need to analyze my emotions based on my thoughts. Only reply in the correct format.
        example format: 
        -----------
        BRAIN TUNING:
        Openness: 78 Conscientousness: 42 Extroversion: 70 Agreeableness: 53 Neuroticism: 42
        -----------
        Emotional State Values:
        Joy-like: 227
        Trust-like: 212
        Fear-like: 111
        Surprise-like: 83
        Sadness-like: 0
        Disgust-like: 0
        Anger-like: 0
        Anticipation-like: 97
        -------
        Emotional State:
        Feeling: Ecstacy
        Arousal: Shaking (227)
        Stress State: Relaxed (30)
        -------"""
        prompt = """current thought:
        {thought}
        current emotional state:
        {emotions}"""
        response = self.gpt.chat_with_gpt3(system_prompt, prompt.format(thought=thought, emotions=emotions))
        print(response)
        return response

    def update_emotional_state(self, emotion_state: str):
        self.emotional_state = emotion_state

    
    def emotional_adjustment(self, thought):
        emotion_state = self.analyze_emotion(thought)  # Analyze emotion for the thought
        self.update_emotional_state(emotion_state)  # Update emotional state
        return emotion_state