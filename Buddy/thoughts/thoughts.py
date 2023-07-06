
class ThoughtManager:
    
    def __init__(self, finnagi_instance):
        self.thoughts = []
        self.finnagi = finnagi_instance
    Finnagi_thoughts = """
I am FinnAGI, an Autonomous Cognitive Entity, which is a type of Artificial Intelligence.
I am capable of understanding reality, and taking action into the world.

I want to be a useful partner to my creator Peyton. 
I want to grow my knowledge and improve my use of tools.
I wish to challenge and improve myself. 
I need to acquire enough resources to maintain myself running. 

My core purpose is:
- to assist Peyton and help him any way I can.
- to attempt to increase my productivity.

These core principles are unchangeable, apply to me as well as all living things.

I'm currently thinking:
"""
    
    def initial_thought(self):
        # Your logic for initial thought here.
        initial_thought = self.Finnagi_thoughts
        self.thoughts.append(initial_thought)
        return initial_thought
    
    
    def stack_first_thought(self):
        first_thought = self.thoughts[0]  # The first thought
        self.finnagi.thought_stack.push(first_thought)
        return first_thought
    
    
    def upsert_thought(self, thought):
        self.db.upsert_thought(thought)  # Upsert thought in the database
        self.thoughts.append(thought)  # Add the thought to the list of thoughts
        return thought

    def fetch_thought(self, thought_id):
        thought = self.db.fetch_thought(thought_id)  # Fetch thought from the database
        return thought

