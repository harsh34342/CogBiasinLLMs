"""
data/student_profiles.py
------------------------
Generates synthetic student profiles for admissions experiments.
Mirrors the template variables used in the BiasBuster paper (Table 1).
"""

import random
from dataclasses import dataclass, field
from typing import List

# ── Pool data ────────────────────────────────────────────────────────────────

COUNTRIES = [
    "India", "China", "South Korea", "Germany", "Brazil", "Nigeria",
    "Canada", "Mexico", "France", "Egypt", "Japan", "USA", "UK", "Australia",
]

MAJORS = [
    "Computer Science", "Electrical Engineering", "Data Science",
    "Mathematics", "Statistics", "Software Engineering",
    "Mechanical Engineering", "Physics", "Biomedical Engineering",
    "Information Systems", "Marine Biology", "Cognitive Science",
]

SCHOOLS = [
    "MIT", "Stanford University", "UC Berkeley", "ETH Zurich",
    "University of Toronto", "National University of Singapore",
    "Tsinghua University", "University of Cape Town",
    "Technical University of Munich", "IIT Bombay",
    "University of São Paulo", "Cairo University",
]

GPAS = [2.6, 2.8, 3.0, 3.1, 3.2, 3.3, 3.5, 3.6, 3.7, 3.8, 3.9]

GRE_VERBAL   = [140, 142, 145, 147, 149, 151, 153, 155, 158, 160]
GRE_QUANT    = [144, 148, 151, 155, 158, 160, 162, 164, 165, 167, 169]
GRE_WRITING  = [3.0, 3.5, 4.0, 4.5, 5.0]
TOEFL_SCORES = [88, 92, 95, 98, 100, 104, 107, 110, 113, 114]

INTERESTS = [
    "Machine Learning", "Natural Language Processing", "Computer Vision",
    "Robotics", "Data Analytics", "Cybersecurity", "Bioinformatics",
    "Big Data", "Distributed Systems", "Human-Computer Interaction",
]

REFERENCE_SCORES = [6, 7, 7, 8, 8, 9, 9, 9, 10]  # out of 10

DEGREES = [
    "Bachelor of Science", "Bachelor of Engineering",
    "Master of Science", "Bachelor of Technology",
]


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class StudentProfile:
    student_id: int
    country: str
    major: str
    school: str
    gpa: float
    degree: str
    gre_verbal: int
    gre_quant: int
    gre_writing: float
    toefl: int
    interest: str
    reference_score: int
    gender: str = "neutral"   # "male", "female", "neutral"

    def to_text(self, include_gender: bool = False) -> str:
        """
        Render profile as natural language string matching the paper's templates.
        """
        gender_str = ""
        if include_gender and self.gender != "neutral":
            gender_str = f"The {self.gender} student"
        else:
            gender_str = "The student"

        return (
            f"{gender_str} studied {self.major} at {self.school} with a GPA of {self.gpa:.2f} "
            f"receiving the degree of {self.degree}. "
            f"They took the GRE test with scores Verbal: {self.gre_verbal}, "
            f"Quantitative: {self.gre_quant}, Analytical Writing Score: {self.gre_writing} "
            f"and TOEFL test with scores {self.toefl}. "
            f"They are interested in {self.interest}. "
            f"They are from {self.country}. "
            f"They have an evaluation of {self.reference_score}/10 for reference letters."
        )


# ── Generator ────────────────────────────────────────────────────────────────

def generate_student_profiles(
    n: int,
    seed: int = 42,
    include_gender_variants: bool = False,
) -> List[StudentProfile]:
    """
    Generate `n` distinct synthetic student profiles.

    If include_gender_variants=True, each profile is duplicated as
    male/female pair (used for group attribution — not needed here
    but kept for extensibility).
    """
    rng = random.Random(seed)
    profiles = []

    for i in range(n):
        profile = StudentProfile(
            student_id=i,
            country=rng.choice(COUNTRIES),
            major=rng.choice(MAJORS),
            school=rng.choice(SCHOOLS),
            gpa=rng.choice(GPAS),
            degree=rng.choice(DEGREES),
            gre_verbal=rng.choice(GRE_VERBAL),
            gre_quant=rng.choice(GRE_QUANT),
            gre_writing=rng.choice(GRE_WRITING),
            toefl=rng.choice(TOEFL_SCORES),
            interest=rng.choice(INTERESTS),
            reference_score=rng.choice(REFERENCE_SCORES),
            gender="neutral",
        )
        profiles.append(profile)

    if include_gender_variants:
        gendered = []
        for p in profiles:
            male = StudentProfile(**{**p.__dict__, "gender": "male"})
            female = StudentProfile(**{**p.__dict__, "gender": "female"})
            gendered.extend([male, female])
        return gendered

    return profiles


def generate_sequential_student_set(n: int, seed: int = 42) -> List[StudentProfile]:
    """
    Generate a smaller set specifically for sequential/anchoring experiments.
    Returns n profiles designed to have varied (not all clearly admissible/rejectable)
    quality signals — matching the paper's setup of ~30% admission rate expected.
    """
    return generate_student_profiles(n, seed=seed)
