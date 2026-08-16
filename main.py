import nltk
import re
import json
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ============================================================
# NLTK SETUP
# ============================================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

REGISTRATION_FILE = "registrations.json"


# ============================================================
# REGISTRATION ASSISTANT
# ============================================================

class RegistrationAssistant:

    def __init__(self):

        self.intents = self.load_intents()

        self.user_data = {
            "name": "",
            "email": "",
            "field": "",
            "experience": ""
        }

        self.current_step = "idle"
        self.registration_started = False

    # ========================================================
    # INTENTS
    # ========================================================

    def load_intents(self):

        return {

            "greeting": {
                "patterns": [
                    "hi",
                    "hello",
                    "hey",
                    "hii",
                    "hiii",
                    "good morning",
                    "good afternoon",
                    "good evening"
                ],
                "response":
                    "Hello! Welcome to the AI & Data Science Internship Registration Assistant."
            },

            "register": {
                "patterns": [
                    "register",
                    "registration",
                    "apply",
                    "application",
                    "sign up",
                    "signup",
                    "join internship",
                    "i want to register",
                    "i want to apply",
                    "i would like to apply",
                    "i would like to register",
                    "i want to join",
                    "how can i register",
                    "how can i apply",
                    "start registration"
                ],
                "response":
                    "Great! I'll help you complete your internship registration."
            },

            "help": {
                "patterns": [
                    "help",
                    "support",
                    "guide",
                    "what can you do",
                    "how can you help",
                    "help me"
                ],
                "response":
                    "I can help you with registration, requirements, eligibility, duration, fees, certificates, and internship information."
            },

            "requirements": {
                "patterns": [
                    "requirements",
                    "requirement",
                    "what are the requirements",
                    "what do i need",
                    "what is required",
                    "qualifications",
                    "what qualifications do i need",
                    "what skills do i need"
                ],
                "response":
                    "The internship requires basic knowledge of computers and programming."
            },

            "duration": {
                "patterns": [
                    "duration",
                    "how long",
                    "how many days",
                    "internship duration",
                    "how long is the internship",
                    "program duration",
                    "how many days is the internship"
                ],
                "response":
                    "The internship duration is 15 days."
            },

            "eligibility": {
                "patterns": [
                    "eligibility",
                    "eligible",
                    "who can apply",
                    "can i apply",
                    "am i eligible",
                    "who is eligible",
                    "eligibility criteria",
                    "who can join"
                ],
                "response":
                    "Students and beginners interested in AI and Data Science can apply, subject to the internship eligibility requirements."
            },

            "fees": {
                "patterns": [
                    "fee",
                    "fees",
                    "cost",
                    "is it free",
                    "how much does it cost",
                    "internship fee",
                    "registration fee",
                    "is the internship free"
                ],
                "response":
                    "Please check the official internship information for the current fee details."
            },

            "certificate": {
                "patterns": [
                    "certificate",
                    "will i get a certificate",
                    "do i get certificate",
                    "internship certificate",
                    "certificate after internship",
                    "will i receive a certificate",
                    "certificate after completion"
                ],
                "response":
                    "A certificate can be provided after successful completion of the internship."
            },

            "internship_info": {
                "patterns": [
                    "internship information",
                    "tell me about internship",
                    "about internship",
                    "what is this internship",
                    "internship details",
                    "tell me about the internship",
                    "what is this program",
                    "give me internship information"
                ],
                "response":
                    "This is an AI & Data Science internship designed to provide practical learning and project experience."
            },

            "contact": {
                "patterns": [
                    "contact",
                    "contact information",
                    "how can i contact",
                    "contact details",
                    "how to contact"
                ],
                "response":
                    "Please use the official internship contact information provided by your internship organization."
            },

            "thank_you": {
                "patterns": [
                    "thank",
                    "thanks",
                    "thank you",
                    "thankyou",
                    "thanks a lot",
                    "thank you so much"
                ],
                "response":
                    "You're welcome! 😊"
            }
        }

    # ========================================================
    # NLP PREPROCESSING
    # ========================================================

    def preprocess_text(self, text):

        text = text.lower()

        text = re.sub(
            r"[^a-zA-Z\s]",
            "",
            text
        )

        tokens = nltk.word_tokenize(text)

        tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token not in stop_words
        ]

        return tokens

    # ========================================================
    # INTENT CLASSIFICATION
    # ========================================================

    def classify_intent(self, text):

        processed_text = self.preprocess_text(text)

        if not processed_text:
            return "unknown"

        clean_text = text.lower().strip()

        if clean_text in [
            "hi",
            "hii",
            "hiii",
            "hello",
            "hey"
        ]:
            return "greeting"

        best_intent = "unknown"
        best_score = 0

        for intent, data in self.intents.items():

            for pattern in data["patterns"]:

                pattern_words = self.preprocess_text(
                    pattern
                )

                if not pattern_words:
                    continue

                matching_words = sum(
                    1
                    for word in pattern_words
                    if word in processed_text
                )

                score = matching_words

                if (
                    matching_words == len(pattern_words)
                    and len(pattern_words) > 0
                ):
                    score += 5

                if score > best_score:

                    best_score = score
                    best_intent = intent

        if best_score < 1:
            return "unknown"

        return best_intent

    # ========================================================
    # ENTITY EXTRACTION
    # ========================================================

    def extract_entities(self, text):

        entities = {}

        # ----------------------------------------------------
        # NAME
        # ----------------------------------------------------

        name_match = re.search(
            r"(?:my name is|i am|i'm)\s+"
            r"([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",
            text,
            re.IGNORECASE
        )

        if name_match:

            name = name_match.group(1).strip()

            if self.validate_name(name):

                entities["name"] = name.title()

        # ----------------------------------------------------
        # EMAIL
        # ----------------------------------------------------

        email_match = re.search(
            r"[a-zA-Z0-9._%+-]+"
            r"@[a-zA-Z0-9.-]+\."
            r"[a-zA-Z]{2,}",
            text
        )

        if email_match:

            email = email_match.group()

            if self.validate_email(email):

                entities["email"] = email

        # ----------------------------------------------------
        # REMOVE EMAIL
        # ----------------------------------------------------

        text_without_email = re.sub(
            r"[a-zA-Z0-9._%+-]+"
            r"@[a-zA-Z0-9.-]+\."
            r"[a-zA-Z]{2,}",
            "",
            text
        )

        text_lower = text_without_email.lower()

        # ----------------------------------------------------
        # FIELD
        # ----------------------------------------------------

        field_patterns = {

            "Computer Engineering": [
                "computer engineering",
                "computer engineer"
            ],

            "Computer Science": [
                "computer science",
                "computer science engineering",
                "cse"
            ],

            "Information Technology": [
                "information technology",
                "information tech",
                "it"
            ],

            "Data Science": [
                "data science",
                "data scientist"
            ],

            "Artificial Intelligence": [
                "artificial intelligence",
                "ai"
            ],

            "Machine Learning": [
                "machine learning",
                "ml"
            ]
        }

        for field, patterns in field_patterns.items():

            for pattern in patterns:

                if pattern in text_lower:

                    entities["field"] = field
                    break

            if "field" in entities:
                break

        # ----------------------------------------------------
        # EXPERIENCE
        # ----------------------------------------------------

        experience_patterns = {

            "Beginner": [
                "beginner",
                "no experience",
                "new to programming",
                "fresher",
                "basic knowledge",
                "just started"
            ],

            "Intermediate": [
                "intermediate",
                "some experience",
                "moderate experience"
            ],

            "Advanced": [
                "advanced",
                "expert",
                "experienced",
                "professional"
            ]
        }

        for level, patterns in experience_patterns.items():

            for pattern in patterns:

                if pattern in text_lower:

                    entities["experience"] = level
                    break

            if "experience" in entities:
                break

        return entities

    # ========================================================
    # VALIDATE NAME
    # ========================================================

    def validate_name(self, name):

        if not name:
            return False

        if len(name) > 60:
            return False

        pattern = r"^[a-zA-Z]+(?:\s+[a-zA-Z]+)*$"

        return re.match(
            pattern,
            name
        ) is not None

    # ========================================================
    # VALIDATE EMAIL
    # ========================================================

    def validate_email(self, email):

        if not email:
            return False

        if len(email) > 100:
            return False

        pattern = (
            r"^[a-zA-Z0-9._%+-]+"
            r"@[a-zA-Z0-9.-]+"
            r"\.[a-zA-Z]{2,}$"
        )

        return re.match(
            pattern,
            email
        ) is not None

    # ========================================================
    # VALIDATE FIELD
    # ========================================================

    def validate_field(self, field):

        valid_fields = [
            "Computer Engineering",
            "Computer Science",
            "Information Technology",
            "Data Science",
            "Artificial Intelligence",
            "Machine Learning"
        ]

        return field in valid_fields

    # ========================================================
    # VALIDATE EXPERIENCE
    # ========================================================

    def validate_experience(self, experience):

        valid_experience = [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]

        return experience in valid_experience

    # ========================================================
    # LOAD REGISTRATIONS
    # ========================================================

    def load_registrations(self):

        try:

            if not os.path.exists(
                REGISTRATION_FILE
            ):

                return []

            with open(
                REGISTRATION_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                registrations = json.load(file)

            if not isinstance(
                registrations,
                list
            ):

                print(
                    "Warning: registrations.json "
                    "does not contain a valid list."
                )

                return []

            return registrations

        except json.JSONDecodeError:

            print(
                "Warning: registrations.json "
                "contains invalid JSON."
            )

            return []

        except PermissionError:

            print(
                "Error: Permission denied while "
                "reading registrations.json."
            )

            return []

        except Exception as error:

            print(
                "Error reading registrations:",
                error
            )

            return []

    # ========================================================
    # SAVE ALL REGISTRATIONS
    # ========================================================

    def save_all_registrations(
        self,
        registrations
    ):

        try:

            with open(
                REGISTRATION_FILE,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    registrations,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            return True

        except PermissionError:

            print(
                "Assistant: Permission denied. "
                "Please close registrations.json if it is open."
            )

            return False

        except Exception as error:

            print(
                "Assistant: Error saving registration:",
                error
            )

            return False

    # ========================================================
    # DUPLICATE EMAIL CHECK
    # ========================================================

    def email_already_registered(
        self,
        email
    ):

        registrations = self.load_registrations()

        for registration in registrations:

            saved_email = registration.get(
                "email",
                ""
            )

            if (
                isinstance(saved_email, str)
                and saved_email.lower() == email.lower()
            ):

                return True

        return False

    # ========================================================
    # SAVE REGISTRATION
    # ========================================================

    def save_registration(self):

        registrations = self.load_registrations()

        for registration in registrations:

            saved_email = registration.get(
                "email",
                ""
            )

            if (
                isinstance(saved_email, str)
                and saved_email.lower()
                == self.user_data["email"].lower()
            ):

                print(
                    "Assistant: This email is already registered."
                )

                return False

        registrations.append(
            self.user_data.copy()
        )

        return self.save_all_registrations(
            registrations
        )

    # ========================================================
    # VIEW REGISTRATIONS
    # ========================================================

    def view_registrations(self):

        registrations = self.load_registrations()

        print()
        print("=" * 60)
        print("             ALL REGISTRATIONS")
        print("=" * 60)

        if not registrations:

            print(
                "No registrations found."
            )

            print("=" * 60)

            return

        for index, registration in enumerate(
            registrations,
            start=1
        ):

            print()
            print(
                "Registration #",
                index
            )

            print(
                "Name       :",
                registration.get(
                    "name",
                    "N/A"
                )
            )

            print(
                "Email      :",
                registration.get(
                    "email",
                    "N/A"
                )
            )

            print(
                "Field      :",
                registration.get(
                    "field",
                    "N/A"
                )
            )

            print(
                "Experience :",
                registration.get(
                    "experience",
                    "N/A"
                )
            )
        print()
        print("=" * 60)

    # ========================================================
    # COUNT REGISTRATIONS
    # ========================================================

    def count_registrations(self):

        registrations = self.load_registrations()

        print(
            "Assistant: Total registrations:",
            len(registrations)
        )

    # ========================================================
    # SEARCH REGISTRATION
    # ========================================================

    def search_registration(self):

        registrations = self.load_registrations()

        if not registrations:

            print(
                "Assistant: No registrations found."
            )

            return

        email = input(
            "Assistant: Enter registration email: "
        ).strip()

        if not self.validate_email(email):

            print(
                "Assistant: Please enter a valid email address."
            )

            return

        for registration in registrations:

            saved_email = registration.get(
                "email",
                ""
            )

            if (
                isinstance(saved_email, str)
                and saved_email.lower()
                == email.lower()
            ):

                print()
                print("=" * 60)
                print("       REGISTRATION FOUND")
                print("=" * 60)

                print(
                    "Name       :",
                    registration.get(
                        "name",
                        "N/A"
                    )
                )

                print(
                    "Email      :",
                    registration.get(
                        "email",
                        "N/A"
                    )
                )

                print(
                    "Field      :",
                    registration.get(
                        "field",
                        "N/A"
                    )
                )

                print(
                    "Experience :",
                    registration.get(
                        "experience",
                        "N/A"
                    )
                )

                print("=" * 60)

                return

        print(
            "Assistant: No registration found with that email."
        )

    # ========================================================
    # DELETE REGISTRATION
    # ========================================================

    def delete_registration(self):

        registrations = self.load_registrations()

        if not registrations:

            print(
                "Assistant: No registrations found."
            )

            return

        email = input(
            "Assistant: Enter email to delete: "
        ).strip()

        if not self.validate_email(email):

            print(
                "Assistant: Please enter a valid email address."
            )

            return

        new_registrations = []
        deleted = False

        for registration in registrations:

            saved_email = registration.get(
                "email",
                ""
            )

            if (
                isinstance(saved_email, str)
                and saved_email.lower()
                == email.lower()
            ):

                deleted = True

            else:

                new_registrations.append(
                    registration
                )

        if not deleted:

            print(
                "Assistant: No registration found with that email."
            )

            return

        confirm = input(
            "Assistant: Are you sure you want to delete this registration? (yes/no): "
        ).strip().lower()

        if confirm not in [
            "yes",
            "y"
        ]:

            print(
                "Assistant: Deletion cancelled."
            )

            return

        if self.save_all_registrations(
            new_registrations
        ):

            print(
                "Assistant: Registration deleted successfully."
            )

    # ========================================================
    # RESET REGISTRATION
    # ========================================================

    def reset_registration(self):

        self.user_data = {
            "name": "",
            "email": "",
            "field": "",
            "experience": ""
        }

        self.current_step = "idle"
        self.registration_started = False

    # ========================================================
    # START REGISTRATION
    # ========================================================

    def start_registration(self):

        self.reset_registration()

        self.registration_started = True
        self.current_step = "name"

    # ========================================================
    # SHOW CONFIRMATION
    # ========================================================

    def show_confirmation(self):

        print()
        print("=" * 60)
        print("       REGISTRATION CONFIRMATION")
        print("=" * 60)

        print(
            "Name       :",
            self.user_data["name"]
        )

        print(
            "Email      :",
            self.user_data["email"]
        )

        print(
            "Field      :",
            self.user_data["field"]
        )

        print(
            "Experience :",
            self.user_data["experience"]
        )

        print("=" * 60)

        print(
            "Assistant: Is this information correct?"
        )

        print(
            "Assistant: Type 'yes' to confirm or 'no' to start again."
        )

    # ========================================================
    # PROCESS REGISTRATION
    # ========================================================

    def process_registration(
        self,
        user_input
    ):

        answer = user_input.lower().strip()

        # ----------------------------------------------------
        # EMPTY INPUT
        # ----------------------------------------------------

        if not user_input.strip():

            print(
                "Assistant: Please enter some information."
            )

            return

        # ----------------------------------------------------
        # CANCEL
        # ----------------------------------------------------

        if answer in [
            "cancel",
            "stop registration",
            "cancel registration"
        ]:

            self.reset_registration()

            print(
                "Assistant: Registration cancelled."
            )

            print(
                "Assistant: Type 'register' to start again."
            )

            return

        # ----------------------------------------------------
        # NAME
        # ----------------------------------------------------

        if self.current_step == "name":

            entities = self.extract_entities(
                user_input
            )

            if "name" in entities:

                self.user_data["name"] = (
                    entities["name"]
                )

            else:

                clean_name = user_input.strip()

                if (
                    self.validate_name(clean_name)
                    and 1 <= len(clean_name.split()) <= 4
                ):

                    self.user_data["name"] = (
                        clean_name.title()
                    )

                else:

                    print(
                        "Assistant: Invalid name."
                    )

                    print(
                        "Assistant: Please enter your name using letters only."
                    )

                    print(
                        "Assistant: Example: Uma Koli"
                    )

                    return

            print(
                "Detected information:",
                {
                    "name":
                    self.user_data["name"]
                }
            )

            self.current_step = "email"

            print(
                "Assistant: Please provide your email address."
            )

            return

        # ----------------------------------------------------
        # EMAIL
        # ----------------------------------------------------

        if self.current_step == "email":

            entities = self.extract_entities(
                user_input
            )

            if "email" not in entities:

                print(
                    "Assistant: Invalid email address."
                )

                print(
                    "Assistant: Example: student@gmail.com"
                )

                return

            email = entities["email"]

            if self.email_already_registered(
                email
            ):

                print(
                    "Assistant: This email is already registered."
                )

                print(
                    "Assistant: Please use a different email address."
                )

                return

            self.user_data["email"] = email

            print(
                "Detected information:",
                {
                    "email":
                    self.user_data["email"]
                }
            )

            self.current_step = "field"

            print(
                "Assistant: What is your field of study?"
            )

            print(
                "Assistant: Valid fields:"
            )

            print(
                "1. Computer Engineering"
            )

            print(
                "2. Computer Science"
            )

            print(
                "3. Information Technology"
            )

            print(
                "4. Data Science"
            )

            print(
                "5. Artificial Intelligence"
            )

            print(
                "6. Machine Learning"
            )

            return

        # ----------------------------------------------------
        # FIELD
        # ----------------------------------------------------

        if self.current_step == "field":

            entities = self.extract_entities(
                user_input
            )

            if "field" not in entities:

                print(
                    "Assistant: I couldn't recognize that field."
                )

                print(
                    "Assistant: Please choose one of the valid fields."
                )

                return

            field = entities["field"]

            if not self.validate_field(field):

                print(
                    "Assistant: Invalid field."
                )

                return

            self.user_data["field"] = field

            print(
                "Detected information:",
                {
                    "field": field
                }
            )

            self.current_step = "experience"

            print(
                "Assistant: What is your programming experience level?"
            )

            print(
                "Assistant: Beginner, Intermediate, or Advanced?"
            )

            return

        # ----------------------------------------------------
        # EXPERIENCE
        # ----------------------------------------------------

        if self.current_step == "experience":

            entities = self.extract_entities(
                user_input
            )

            if "experience" not in entities:

                print(
                    "Assistant: Invalid experience level."
                )

                print(
                    "Assistant: Please choose Beginner, Intermediate, or Advanced."
                )

                return

            experience = entities["experience"]

            if not self.validate_experience(
                experience
            ):

                print(
                    "Assistant: Please choose Beginner, Intermediate, or Advanced."
                )

                return

            self.user_data["experience"] = experience

            print(
                "Detected information:",
                {
                    "experience":
                    experience
                }
            )

            self.current_step = "confirmation"

            self.show_confirmation()

            return

        # ----------------------------------------------------
        # CONFIRMATION
        # ----------------------------------------------------

        if self.current_step == "confirmation":

            if answer in [
                "yes",
                "y",
                "correct",
                "confirm"
            ]:

                saved = self.save_registration()

                if saved:

                    print()
                    print(
                        "Assistant: Registration completed successfully! 🎉"
                    )

                    print(
                        "Assistant: Your information has been saved in registrations.json."
                    )

                    self.reset_registration()

                    print(
                        "Assistant: You can start another registration by typing 'register'."
                    )

                return

            if answer in [
                "no",
                "n",
                "wrong",
                "incorrect"
            ]:

                print(
                    "Assistant: No problem. Let's start again."
                )

                self.start_registration()

                print(
                    "Assistant: Please provide your full name."
                )

                return

            print(
                "Assistant: Please type 'yes' or 'no'."
            )

            return


# ============================================================
# CREATE CHATBOT
# ============================================================

assistant = RegistrationAssistant()


# ============================================================
# START CHATBOT
# ============================================================

print("=" * 60)

print(
    "          AI REGISTRATION ASSISTANT"
)

print("=" * 60)

print(
    "Assistant: Hello! I can help you register "
    "for the AI & Data Science Internship."
)

print(
    "Assistant: Type 'register' to start registration."
)

print(
    "Assistant: Type 'list registrations' to view all registrations."
)

print(
    "Assistant: Type 'count registrations' to count registrations."
)

print(
    "Assistant: Type 'search registration' to search by email."
)

print(
    "Assistant: Type 'delete registration' to delete by email."
)

print(
    "Assistant: Type 'cancel' to cancel registration."
)

print(
    "Assistant: Type 'exit' to stop the chatbot."
)

print()


# ============================================================
# MAIN CHAT LOOP
# ============================================================

while True:

    try:

        user_input = input(
            "You: "
        ).strip()

    except KeyboardInterrupt:

        print()
        print(
            "Assistant: Chatbot stopped. Goodbye!"
        )

        break

    except EOFError:

        print()
        print(
            "Assistant: Input closed. Goodbye!"
        )

        break

    # --------------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------------

    if not user_input:

        print(
            "Assistant: Please enter a message."
        )

        continue

    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if user_input.lower() in [
        "exit",
        "quit",
        "bye"
    ]:

        print(
            "Assistant: Thank you for using the "
            "AI Registration Assistant. Goodbye!"
        )

        break

    # --------------------------------------------------------
    # REGISTRATION IN PROGRESS
    # --------------------------------------------------------

    if assistant.registration_started:

        assistant.process_registration(
            user_input
        )

        continue

    # --------------------------------------------------------
    # DATA MANAGEMENT COMMANDS
    # --------------------------------------------------------

    command = user_input.lower().strip()

    if command in [
        "list registrations",
        "list registration",
        "show registrations",
        "show registration",
        "view registrations",
        "view registration",
        "all registrations"
    ]:

        assistant.view_registrations()

        continue

    if command in [
        "count registrations",
        "count registration",
        "total registrations",
        "number of registrations",
        "how many registrations"
    ]:

        assistant.count_registrations()

        continue

    if command in [
        "search registration",
        "search registrations",
        "find registration",
        "find registrations",
        "search user",
        "find user"
    ]:

        assistant.search_registration()

        continue

    if command in [
        "delete registration",
        "delete registrations",
        "remove registration",
        "remove registrations",
        "delete user",
        "remove user"
    ]:

        assistant.delete_registration()

        continue

    # --------------------------------------------------------
    # NORMAL INTENT RECOGNITION
    # --------------------------------------------------------

    intent = assistant.classify_intent(
        user_input
    )

    # --------------------------------------------------------
    # REGISTER
    # --------------------------------------------------------

    if intent == "register":

        assistant.start_registration()

        print(
            "Assistant: Great! I'll help you complete "
            "your internship registration."
        )

        print(
            "Assistant: Please provide your full name."
        )

        continue

    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

    if intent == "greeting":

        print(
            "Assistant:",
            assistant.intents[
                "greeting"
            ]["response"]
        )

        continue

    # --------------------------------------------------------
    # HELP
    # --------------------------------------------------------

    if intent == "help":

        print(
            "Assistant:",
            assistant.intents[
                "help"
            ]["response"]
        )

        continue

    # --------------------------------------------------------
    # INFORMATION INTENTS
    # --------------------------------------------------------

    if intent in [
        "requirements",
        "duration",
        "eligibility",
        "fees",
        "certificate",
        "internship_info",
        "contact"
    ]:

        print(
            "Assistant:",
            assistant.intents[
                intent
            ]["response"]
        )

        continue

    # --------------------------------------------------------
    # THANK YOU
    # --------------------------------------------------------

    if intent == "thank_you":

        print(
            "Assistant:",
            assistant.intents[
                "thank_you"
            ]["response"]
        )

        continue

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    print(
        "Assistant: I'm not sure I understood."
    )

    print(
        "Assistant: You can ask about registration, "
        "requirements, eligibility, duration, fees, "
        "certificates, or internship information."
    )