class_names_5 = [
    "Neutral (student in class).",
    "Enjoyment (student in class).",
    "Confusion (student in class).",
    "Fatigue (student in class).",
    "Distraction (student in class)."
]

class_names_with_context_5 = [
    "A student shows a neutral learning state in a classroom.",
    "A student shows enjoyment while learning in a classroom.",
    "A student shows confusion during learning in a classroom.",
    "A student shows fatigue during learning in a classroom.",
    "A student shows distraction and is not focused in a classroom."
]

class_descriptor_5_only_face = [
    "A student has a neutral face with relaxed mouth, open eyes, and calm eyebrows.",
    "A student looks happy with a slight smile, bright eyes, and relaxed eyebrows.",
    "A student looks confused with furrowed eyebrows, a puzzled look, and slightly open mouth.",
    "A student looks tired with drooping eyelids, frequent yawning, and a sleepy face.",
    "A student looks distracted with unfocused eyes and a wandering gaze away from the lesson."
]

class_descriptor_5_only_body = [
    "A student sits still with an upright posture and hands on the desk, showing a neutral learning state.",
    "A student leans slightly forward with an open, engaged posture, showing enjoyment in learning.",
    "A student tilts the head and leans in, hand on chin, showing confusion while trying to understand.",
    "A student slouches with shoulders dropped and head lowered, showing fatigue during class.",
    "A student shifts around, turns away from the desk, or looks sideways, showing distraction and low focus."
]

class_descriptor_5 = [
    "A student looks neutral and calm in class, with a relaxed face and steady gaze, quietly watching the lecture or reading notes.",
    "A student shows enjoyment while learning, with a gentle smile and bright eyes, appearing engaged and interested in the lesson.",
    "A student looks confused in class, with furrowed eyebrows and a puzzled expression, focusing on the material as if trying to understand.",
    "A student appears fatigued in class, with drooping eyelids and yawning, head slightly lowered, showing low energy.",
    "A student is distracted in class, frequently looking away from the lesson, scanning around, and not paying attention to learning materials."
]

# Prompt Ensemble for RAER (5 classes)
# Each inner list contains multiple descriptions for a single class.
prompt_ensemble_5 = [
    [   # Neutral
        "A photo of a student with a neutral expression.",
        "A photo of a student sitting still and watching the lecture.",
        "A photo of a student with a calm face and neutral body posture."
    ],
    [   # Enjoyment
        "A photo of a student showing enjoyment while learning.",
        "A photo of a student with a happy face and a slight smile.",
        "A photo of a student who looks engaged and interested in the lesson."
    ],
    [   # Confusion
        "A photo of a student who is confused.",
        "A photo of a student with a puzzled look and furrowed eyebrows.",
        "A photo of a student staring at the material as if trying to understand."
    ],
    [   # Fatigue
        "A photo of a student who appears fatigued or sleepy.",
        "A photo of a student with drooping eyelids or yawning.",
        "A photo of a student showing low energy with a lowered head."
    ],
    [   # Distraction
        "A photo of a student who is distracted from learning.",
        "A photo of a student looking away from the lesson or checking a phone.",
        "A photo of a student with a wandering gaze and unfocused eyes."
    ]
]

# ==========================================
# DAISEE Definitions (4 levels of Engagement)
# 0: Very Low, 1: Low, 2: High, 3: Very High
# ==========================================

class_names_daisee = [
    "Very Low Engagement",
    "Low Engagement",
    "High Engagement",
    "Very High Engagement"
]

class_names_with_context_daisee = [
    "A person shows very low engagement and is completely uninterested or asleep.",
    "A person shows low engagement and seems bored, tired or distracted.",
    "A person shows high engagement and is calmly paying attention to the screen.",
    "A person shows very high engagement and is actively reacting and focused."
]

class_descriptor_daisee = [
    "A person is looking away, has eyes closed, head down, or is completely ignoring the screen.",
    "A person is yawning, scratching face, looking around, or resting head on hand, showing boredom.",
    "A person is sitting upright, looking directly at the screen, with a neutral but attentive face.",
    "A person is leaning forward, smiling, talking, or taking notes, showing intense interest."
]

# Prompt Ensemble for DAISEE (Engagement Levels)
# REFINED FOR WEBCAM CONTEXT & DISTINGUISHABILITY
prompt_ensemble_daisee = [
    [   # Level 0: Very Low (Avoidance / Sleep)
        "A webcam view of a person looking completely away from the computer screen.",
        "A close-up of a person with eyes closed, sleeping, or head down on the desk.",
        "A photo of a person talking to someone else in the room, ignoring the computer.",
        "A webcam shot of a person leaving the chair or turning their back to the camera.",
        "A face of a person showing zero interest, completely disengaged from online learning."
    ],
    [   # Level 1: Low (Passive Distraction / Boredom)
        "A webcam view of a person looking bored, yawning, or rubbing their eyes.",
        "A photo of a person resting their chin on their hand, looking tired and blank.",
        "A person fidgeting, scratching their head, or looking around the room distractedly.",
        "A face showing signs of fatigue, drooping eyelids, and lack of energy.",
        "A student checking their phone or multitasking instead of focusing on the screen."
    ],
    [   # Level 2: High (Steady / Passive Focus)
        "A webcam view of a person looking directly and steadily at the computer screen.",
        "A photo of a person sitting upright with a neutral, calm, and attentive face.",
        "A face of a student quietly watching the online lecture without much movement.",
        "A person maintaining consistent eye contact with the monitor, showing normal focus.",
        "A student listening to the lesson with a serious and composed expression."
    ],
    [   # Level 3: Very High (Active / Intense Focus)
        "A webcam view of a person leaning forward towards the screen with intense focus.",
        "A photo of a student nodding, smiling, or mouthing words in response to the lesson.",
        "A face showing excitement, surprise, or deep cognitive engagement with the content.",
        "A person taking notes vigorously or looking very sharp and alert.",
        "A student showing high energy and strong positive reaction to the online class."
    ]
]
