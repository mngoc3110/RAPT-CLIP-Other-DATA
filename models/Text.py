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
# UPGRADE: 5 Distinctive Prompts per class (SOTA Plan)
prompt_ensemble_daisee = [
    [   # Level 0: Very Low (Total Disengagement)
        "A photo of a person looking away from the screen completely.",
        "A photo of a person with eyes closed, sleeping or dozing off.",
        "A photo of a person with their head down on the desk.",
        "A photo of a person ignoring the lesson and looking sideways.",
        "A photo of a person showing zero interest and completely disengaged."
    ],
    [   # Level 1: Low (Boredom/Fatigue)
        "A photo of a person yawning or looking extremely tired.",
        "A photo of a person resting their head on their hand in boredom.",
        "A photo of a person fidgeting, scratching their face or hair.",
        "A photo of a person looking around the room distractedly.",
        "A photo of a person shifting their body posture frequently."
    ],
    [   # Level 2: High (Steady Attention)
        "A photo of a person looking steadily and directly at the screen.",
        "A photo of a person sitting upright with a calm, neutral face.",
        "A photo of a person paying attention to the video content.",
        "A photo of a person maintaining eye contact with the screen.",
        "A photo of a person showing a normal, focused learning state."
    ],
    [   # Level 3: Very High (Active Interaction)
        "A photo of a person leaning forward with intense focus.",
        "A photo of a person smiling, laughing, or reacting to the video.",
        "A photo of a person writing notes, talking, or nodding actively.",
        "A photo of a person showing high energy and excitement.",
        "A photo of a person who is deeply engrossed in the lecture."
    ]
]
