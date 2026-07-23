from voice_assistant import listen, speak

print("Starting test...")

command = listen()

print("Result:", command)

speak("I heard " + command)