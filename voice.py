import pyttsx3

def speak(text):
    """
    This is the updated version that prevents the 'run loop' error.
    """
    try:
        
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        
    except Exception as e:
        print(f"Speech error: {e}")