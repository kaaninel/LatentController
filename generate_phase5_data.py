"""
Generate Phase 5 synthetic training dataset.

Produces a diverse, adult-level dataset designed to:
  1. Force memory dependence (made-up entities that can't be in weights)
  2. Teach absorption vs emission (NOOP targets for context, real targets for answers)
  3. Push beyond TinyStories vocabulary into real English
  4. Cover multiple domains: science, geography, people, events, instructions, conversation

Also produces a knowledge base (JSONL) for memory pre-seeding.

Usage:
    python generate_phase5_data.py                          # default 150K sequences
    python generate_phase5_data.py --total_qa 80000         # custom QA count
    python generate_phase5_data.py --no_tinystories         # skip replay
"""

import os
import json
import argparse
import random
from typing import Tuple, List

import torch
from datasets import load_dataset

from tokenizer_utils import load_tokenizer, encode


# ==========================================================================
# Entity pools — all FICTIONAL to ensure answers can't come from weights
# ==========================================================================

FIRST_NAMES = [
    "Amara", "Kai", "Linnea", "Rohan", "Celine", "Dmitri", "Ines", "Tariq",
    "Yuki", "Soren", "Priya", "Mateo", "Freya", "Hassan", "Lena", "Orin",
    "Zara", "Theo", "Maren", "Idris", "Elara", "Nico", "Saida", "Leif",
    "Nadia", "Callum", "Asha", "Rafael", "Ingrid", "Dante", "Vera", "Kiran",
    "Petra", "Emeka", "Solene", "Arjun", "Hana", "Felix", "Rosa", "Eamon",
    "Talia", "Jovan", "Mira", "Stellan", "Anika", "Bram", "Isla", "Reza",
    "Maeve", "Lukas", "Yara", "Owen", "Faye", "Marcus", "Dara", "Silas",
    "Nina", "Erik", "Lira", "Hugo", "Thea", "Nils", "Ivy", "Leon",
]

LAST_NAMES = [
    "Chen", "Erikson", "Okafor", "Petrov", "Vasquez", "Hartwell", "Nakamura",
    "Brennan", "Adesanya", "Lindqvist", "Sharma", "Delacroix", "Mbeki",
    "Kowalski", "Tanaka", "Reeves", "Johansson", "Otieno", "Marek", "Farouk",
    "Strand", "Alvarez", "Nyberg", "Kapoor", "Thornton", "Voss", "Okoro",
    "Larsen", "Dubois", "Kimura", "Ashford", "Rao", "Bergmann", "Salazar",
    "Wren", "Nkosi", "Calloway", "Ishida", "Moreau", "Hagen",
]

ORGANIZATIONS = [
    "Meridian Research Institute", "Northlight Dynamics", "Cascade Applied Sciences",
    "Solaris Engineering Group", "Vantage Analytics", "Helix Biotech",
    "Arclight Laboratories", "Summit Climate Research", "Beacon Aerospace",
    "Thornfield Consulting", "Prism Data Systems", "Ridgeline Robotics",
    "Atlas Materials Lab", "Nexus Quantum Computing", "Verdant Energy Solutions",
    "Crosswind Maritime", "Sterling Optics", "Ironforge Manufacturing",
    "Palisade Defense Research", "Keystone AI Group", "Silverleaf Pharma",
    "Redwood Structural", "Halcyon Renewables", "Vanguard Space Systems",
    "Obsidian Security", "Brightwater Institute", "Cobalt Medical Devices",
    "Pinecrest University", "Harbor Point Analytics", "Whitestone Foundation",
]

LOCATIONS = [
    "Thornfield Bay", "Ashford Crossing", "Millhaven", "Cedar Point",
    "Ridgewater", "Brackenmoor", "Stonebridge", "Lakeport",
    "Cliffdale", "Windhaven", "Harborfield", "Maplecrest",
    "Hollowbrook", "Driftwood Cove", "Ironside", "Clearwater Falls",
    "Pineridge", "Saltmarsh", "Foxglove", "Westmere",
    "Highgate", "Moorfield", "Ravensdale", "Sunridge",
    "Blackstone", "Oldbury", "Ferndale", "Copperhill",
    "Northbank", "Greystone", "Brightvale", "Dunmore",
]

COUNTRIES = [
    "Aldria", "Belvaren", "Calistan", "Dunmere", "Estoria", "Faeland",
    "Galdori", "Helvara", "Istoria", "Jorvik", "Keldara", "Lemuria",
    "Marvane", "Norath", "Olvera", "Palladia", "Ravensland", "Silvane",
    "Thaloria", "Valdris",
]

ROLES = [
    "materials engineer", "climate researcher", "computational biologist",
    "structural analyst", "data architect", "marine acoustician",
    "quantum physicist", "systems designer", "renewable energy specialist",
    "robotics engineer", "aerospace technician", "biomedical researcher",
    "network architect", "process chemist", "geophysicist",
    "urban planner", "cognitive scientist", "firmware developer",
    "environmental analyst", "cryptographer", "logistics coordinator",
    "metallurgist", "astrophysicist", "epidemiologist",
]

PROJECTS = [
    "Project Horizon", "the Vanguard Initiative", "Operation Clearwater",
    "the Atlas Protocol", "Project Lighthouse", "the Meridian Study",
    "the Phoenix Framework", "Project Nightfall", "the Keystone Program",
    "the Halcyon Experiment", "Project Ironclad", "the Summit Array",
    "the Cascade Network", "Project Solstice", "the Beacon Survey",
    "the Prism Architecture", "Project Windbreak", "the Nexus Grid",
]

MATERIALS = [
    "Belmont alloy", "ferrocrystalline composite", "aerogel-titanium matrix",
    "graphene-boron lattice", "ceramo-metallic substrate", "quantum dot film",
    "piezoplastic membrane", "magnetorheological fluid", "nano-ceramic coating",
    "biosynthetic polymer", "cryogenic superconductor blend", "carbon-silicene mesh",
    "hexagonal boron nitride sheet", "photovoltaic crystal array", "thermochromic gel",
]

DEVICES = [
    "the QuietWave sensor", "the Prism-7 detector", "the ArcLight scanner",
    "the Meridian probe", "the Helix sequencer", "the Vortex analyzer",
    "the Beacon transponder", "the Cascade filter", "the Atlas calibrator",
    "the Solaris collector", "the Nexus switch", "the Pinnacle relay",
    "the Crosswind turbine", "the Halcyon monitor", "the Ironclad shield",
]

JOURNALS = [
    "the Journal of Applied Thermodynamics", "Structural Innovation Quarterly",
    "Advances in Computational Biology", "the Review of Marine Engineering",
    "the International Journal of Quantum Materials",
    "Frontiers in Renewable Systems", "the Annals of Geophysical Research",
    "Applied Robotics Letters", "the Journal of Network Architecture",
    "Cognitive Systems Review", "the Bulletin of Environmental Science",
]

YEARS = list(range(1960, 2026))

NUMBERS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 15: "fifteen", 20: "twenty",
    25: "twenty-five", 30: "thirty", 50: "fifty", 100: "a hundred",
}

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

COLORS = [
    "crimson", "cobalt", "emerald", "amber", "ivory", "charcoal",
    "teal", "bronze", "slate", "copper", "violet", "rust",
]


# ==========================================================================
# Helper functions
# ==========================================================================

def rname() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def rorg() -> str:
    return random.choice(ORGANIZATIONS)

def rloc() -> str:
    return random.choice(LOCATIONS)

def rcountry() -> str:
    return random.choice(COUNTRIES)

def rrole() -> str:
    return random.choice(ROLES)

def rproject() -> str:
    return random.choice(PROJECTS)

def rmaterial() -> str:
    return random.choice(MATERIALS)

def rdevice() -> str:
    return random.choice(DEVICES)

def rjournal() -> str:
    return random.choice(JOURNALS)

def ryear() -> int:
    return random.choice(YEARS)

def rmonth() -> str:
    return random.choice(MONTHS)

def rcolor() -> str:
    return random.choice(COLORS)

def rnumber() -> Tuple[int, str]:
    n = random.choice(list(NUMBERS.keys()))
    return n, NUMBERS[n]

def pick(pool, n=1, exclude=None):
    available = [x for x in pool if x != exclude] if exclude else pool
    if n == 1:
        return random.choice(available)
    return random.sample(available, min(n, len(available)))


# ==========================================================================
# QA Generators — each returns (context: str, answer: str)
# Context includes the passage AND the question.
# Answer is a short span (typically 1-15 words).
# ==========================================================================

def gen_person_role() -> Tuple[str, str]:
    """Person works at organization in a role."""
    name = rname()
    role = rrole()
    org = rorg()
    loc = rloc()
    year = ryear()

    passages = [
        f"{name} has worked as a {role} at {org} since {year}. The institute is based in {loc}.",
        f"Based in {loc}, {org} appointed {name} as their lead {role} in {year}.",
        f"Since {year}, {name} has held the position of {role} at {org}, located in {loc}.",
        f"{org} in {loc} hired {name} in {year}. {name} serves as the department's {role}.",
    ]
    passage = random.choice(passages)

    qa_options = [
        (f"What role does {name} hold?", role),
        (f"Where is {org} based?", loc),
        (f"When did {name} join {org}?", str(year)),
        (f"Who works as a {role} at {org}?", name),
    ]
    question, answer = random.choice(qa_options)
    return f"{passage}\n{question}", answer


def gen_project_details() -> Tuple[str, str]:
    """Project at organization with details."""
    name = rname()
    org = rorg()
    project = rproject()
    year = ryear()
    n, n_word = rnumber()

    passages = [
        f"{org} launched {project} in {year} under the direction of {name}. The project involved a team of {n_word} researchers.",
        f"In {year}, {name} led {project} at {org}. The initiative brought together {n_word} specialists from across the organization.",
        f"{project}, directed by {name} at {org}, began in {year} with a core team of {n_word} members.",
    ]
    passage = random.choice(passages)

    qa_options = [
        (f"Who directed {project}?", name),
        (f"When did {project} begin?", str(year)),
        (f"How many researchers worked on {project}?", n_word),
        (f"Which organization ran {project}?", org),
    ]
    question, answer = random.choice(qa_options)
    return f"{passage}\n{question}", answer


def gen_material_science() -> Tuple[str, str]:
    """Material properties and applications."""
    material = rmaterial()
    org = rorg()
    name = rname()
    year = ryear()
    temp = random.choice(["350", "420", "580", "650", "800", "950", "1100", "1250"])
    strength = random.choice(["twice", "three times", "four times", "five times"])
    application = random.choice([
        "aerospace structural components", "deep-sea pressure vessels",
        "high-temperature turbine blades", "medical implant coatings",
        "battery electrode substrates", "satellite thermal shields",
        "bridge reinforcement cables", "reactor containment walls",
    ])

    passages = [
        f"The {material}, developed at {org} by {name} in {year}, withstands temperatures up to {temp} degrees Celsius. It is {strength} stronger than conventional steel and primarily used in {application}.",
        f"{name} at {org} synthesized the {material} in {year}. With a thermal tolerance of {temp}°C and {strength} the tensile strength of standard alloys, the material has found use in {application}.",
        f"Published in {year}, {name}'s work at {org} introduced the {material}. It tolerates {temp}°C, offers {strength} the strength of traditional composites, and targets {application}.",
    ]
    passage = random.choice(passages)

    qa_options = [
        (f"What temperature can the {material} withstand?", f"{temp} degrees"),
        (f"Who developed the {material}?", name),
        (f"What is the {material} used for?", application),
        (f"How strong is the {material} compared to steel?", f"{strength} stronger"),
        (f"When was the {material} created?", str(year)),
    ]
    question, answer = random.choice(qa_options)
    return f"{passage}\n{question}", answer


def gen_geography() -> Tuple[str, str]:
    """Place descriptions with facts."""
    loc = rloc()
    country = rcountry()
    pop = random.choice(["12,000", "35,000", "48,000", "72,000", "95,000", "120,000", "180,000"])
    feature = random.choice([
        "a natural harbor protected by granite cliffs",
        "an ancient stone bridge spanning the central river",
        "thermal springs along the northern hillside",
        "a dense old-growth forest covering the eastern slopes",
        "a network of limestone caves beneath the town center",
        "a tidal estuary rich in migratory bird species",
        "volcanic soil that produces exceptional vineyards",
        "a freshwater aquifer supplying the entire region",
    ])
    industry = random.choice([
        "sustainable fishing", "precision manufacturing",
        "geothermal energy production", "rare earth mineral extraction",
        "commercial shipping", "agricultural research",
        "timber processing", "pharmaceutical production",
    ])
    event = random.choice([
        "the Annual Harvest Festival", "the Maritime Heritage Week",
        "the International Science Symposium", "the Regional Trade Fair",
        "the Founders Day Celebration", "the Winter Solstice Market",
    ])
    month = rmonth()

    passages = [
        f"{loc} is a town in {country} with a population of roughly {pop}. It is known for {feature}. The local economy depends on {industry}. Every {month}, the town hosts {event}.",
        f"Located in {country}, {loc} (population {pop}) sits beside {feature}. The town's main industry is {industry}, and it draws visitors each {month} for {event}.",
        f"The town of {loc}, {country}, is home to about {pop} people. Its distinguishing feature is {feature}. {industry.capitalize()} drives the economy, while {event} in {month} is a major cultural draw.",
    ]
    passage = random.choice(passages)

    qa_options = [
        (f"What country is {loc} in?", country),
        (f"What is the population of {loc}?", pop),
        (f"What is {loc} known for?", feature),
        (f"What industry drives {loc}'s economy?", industry),
        (f"When does {loc} host {event}?", month),
    ]
    question, answer = random.choice(qa_options)
    return f"{passage}\n{question}", answer


def gen_device_specs() -> Tuple[str, str]:
    """Technical device specifications."""
    device = rdevice()
    org = rorg()
    name = rname()
    year = ryear()
    weight = random.choice(["1.2 kg", "3.5 kg", "7.8 kg", "12 kg", "0.8 kg", "450 grams"])
    range_val = random.choice(["200 meters", "500 meters", "2 kilometers", "5 kilometers", "15 kilometers"])
    power = random.choice(["a lithium-polymer cell", "solar panels", "a miniature fuel cell", "wireless induction", "a rechargeable nickel-zinc battery"])
    freq = random.choice(["12 kHz", "40 kHz", "120 MHz", "2.4 GHz", "5.8 GHz", "900 MHz"])

    passages = [
        f"{device}, designed by {name} at {org} in {year}, weighs {weight} and operates at {freq}. It is powered by {power} and has an effective range of {range_val}.",
        f"Weighing just {weight}, {device} was introduced by {org} in {year}. Developed by {name}, it transmits at {freq} over a range of {range_val}, drawing power from {power}.",
        f"In {year}, {name} at {org} unveiled {device}. Operating at {freq} with a range of {range_val}, the {weight} unit runs on {power}.",
    ]
    passage = random.choice(passages)

    qa_options = [
        (f"How much does {device} weigh?", weight),
        (f"What is the range of {device}?", range_val),
        (f"Who designed {device}?", name),
        (f"What frequency does {device} operate at?", freq),
        (f"How is {device} powered?", power),
    ]
    question, answer = random.choice(qa_options)
    return f"{passage}\n{question}", answer


def gen_publication() -> Tuple[str, str]:
    """Research publication details."""
    name = rname()
    coauthor = rname()
    org = rorg()
    journal = rjournal()
    year = ryear()
    topic = random.choice([
        "cortical plasticity in adult primates",
        "efficient hydrogen storage in metal-organic frameworks",
        "ocean current modeling under polar ice sheets",
        "fault-tolerant quantum error correction codes",
        "antibiotic resistance pathways in soil bacteria",
        "load distribution in cable-stayed bridges",
        "neural signal decoding for prosthetic control",
        "atmospheric methane tracking via satellite imagery",
        "autonomous navigation in GPS-denied environments",
        "protein folding dynamics at extreme pressures",
    ])
    finding = random.choice([
        "a 40% improvement in detection accuracy",
        "a novel three-stage synthesis process",
        "previously unobserved phase transitions",
        "a computationally efficient approximation algorithm",
        "strong correlation between two previously unlinked variables",
        "a self-correcting feedback mechanism",
        "significant reduction in energy consumption",
        "a reproducible method for large-scale fabrication",
    ])

    passages = [
        f"{name} and {coauthor} at {org} published a study on {topic} in {journal} in {year}. Their key finding was {finding}.",
        f"A {year} paper in {journal} by {name} and {coauthor} ({org}) investigated {topic}. The study reported {finding}.",
        f"Researchers {name} and {coauthor} from {org} contributed a {year} article to {journal} examining {topic}, demonstrating {finding}.",
    ]
    passage = random.choice(passages)

    qa_options = [
        (f"Where was {name}'s study on {topic} published?", journal),
        (f"What was the key finding of {name}'s research?", finding),
        (f"Who co-authored the paper with {name}?", coauthor),
        (f"When was the study on {topic} published?", str(year)),
        (f"What institution are {name} and {coauthor} affiliated with?", org),
    ]
    question, answer = random.choice(qa_options)
    return f"{passage}\n{question}", answer


def gen_event_history() -> Tuple[str, str]:
    """Historical events (fictional)."""
    loc = rloc()
    country = rcountry()
    year = ryear()
    name1 = rname()
    name2 = rname()
    event_type = random.choice([
        "trade agreement", "research accord", "infrastructure treaty",
        "maritime boundary settlement", "environmental pact",
        "scientific cooperation framework", "cultural exchange program",
    ])
    duration = random.choice(["a decade-long", "a twenty-year", "a thirty-year", "a five-year", "a century-old"])
    dispute = random.choice([
        "dispute over fishing rights", "disagreement on tariff regulations",
        "conflict over water access", "territorial boundary question",
        "debate about mining permits", "standoff over transit routes",
    ])
    outcome = random.choice([
        "established a joint oversight commission",
        "created a shared economic zone",
        "set up a bilateral research fund",
        "introduced standardized regulations across both regions",
        "opened three new cross-border transit corridors",
        "founded an international monitoring agency",
    ])

    passages = [
        f"The {loc} {event_type}, signed by {name1} and {name2} in {year}, ended {duration} {dispute} between {country} and its northern neighbor. The agreement {outcome}.",
        f"In {year}, delegates {name1} and {name2} signed the {loc} {event_type}, resolving {duration} {dispute} involving {country}. As a result, the treaty {outcome}.",
        f"{name1} and {name2} brokered the {loc} {event_type} in {year} to settle {duration} {dispute} in {country}. The settlement {outcome}.",
    ]
    passage = random.choice(passages)

    qa_options = [
        (f"When was the {loc} {event_type} signed?", str(year)),
        (f"Who signed the {loc} {event_type}?", f"{name1} and {name2}"),
        (f"What did the {loc} {event_type} resolve?", f"{duration} {dispute}"),
        (f"What did the {loc} agreement establish?", outcome),
    ]
    question, answer = random.choice(qa_options)
    return f"{passage}\n{question}", answer


def gen_instruction() -> Tuple[str, str]:
    """Procedural instructions."""
    device = rdevice()
    step1 = random.choice([
        "disconnect the power supply",
        "remove the outer casing",
        "ensure the ambient temperature is below 30°C",
        "verify the firmware version is 4.2 or later",
        "clear the error log from the previous session",
        "attach the calibration module to port B",
    ])
    step2 = random.choice([
        "rotate the alignment dial clockwise until the green indicator lights up",
        "press and hold the reset button for five seconds",
        "insert the test cartridge into the upper slot",
        "run the self-diagnostic routine from the control panel",
        "connect the reference signal cable to the input terminal",
        "adjust the gain knob until the reading stabilizes",
    ])
    step3 = random.choice([
        "wait for the confirmation tone before proceeding",
        "record the displayed baseline value in the maintenance log",
        "replace the outer casing and restore power",
        "verify the output matches the reference within 2% tolerance",
        "upload the calibration data to the central database",
        "restart the device and check for error codes",
    ])
    warning = random.choice([
        "Never attempt this procedure while the device is powered on.",
        "Wear anti-static gloves during the entire process.",
        "The calibration is void if ambient humidity exceeds 60%.",
        "This procedure must be performed by certified technicians only.",
        "Failure to follow these steps in order may damage the sensor array.",
    ])

    passages = [
        f"To calibrate {device}, follow these steps. First, {step1}. Second, {step2}. Third, {step3}. Note: {warning}",
        f"Calibration procedure for {device}: (1) {step1.capitalize()}. (2) {step2.capitalize()}. (3) {step3.capitalize()}. Important: {warning}",
        f"{device} calibration requires three steps. Begin by: {step1}. Then {step2}. Finally, {step3}. Warning: {warning}",
    ]
    passage = random.choice(passages)

    qa_options = [
        (f"What is the first step when calibrating {device}?", step1),
        (f"What should you do after {step1}?", step2),
        (f"What is the final calibration step for {device}?", step3),
        (f"What warning is given about calibrating {device}?", warning),
    ]
    question, answer = random.choice(qa_options)
    return f"{passage}\n{question}", answer


def gen_conversation() -> Tuple[str, str]:
    """Multi-turn conversation with information exchange."""
    name1 = pick(FIRST_NAMES)
    name2 = pick(FIRST_NAMES, exclude=name1)

    templates = [
        # Template A: Meeting logistics
        lambda: _conv_meeting(name1, name2),
        # Template B: Project update
        lambda: _conv_project(name1, name2),
        # Template C: Travel plans
        lambda: _conv_travel(name1, name2),
        # Template D: Technical discussion
        lambda: _conv_technical(name1, name2),
    ]
    return random.choice(templates)()


def _conv_meeting(n1, n2):
    room = random.choice(["Room 4B", "the East Wing conference room", "Lab 7", "the third floor boardroom", "Building C"])
    time = random.choice(["9 AM", "10:30 AM", "2 PM", "3:15 PM", "4 PM", "11 AM"])
    topic = random.choice(["the quarterly review", "the prototype demo", "the budget allocation", "the hiring plan", "the safety audit"])

    passage = f'{n1}: The meeting on {topic} is moved to {time} tomorrow.\n{n2}: Where is it being held?\n{n1}: {room}. Bring the latest figures.\n{n2}: Got it. Should I also prepare the cost breakdown?\n{n1}: Yes, and the timeline update too.'

    qa_options = [
        (f"What time is the meeting?", time),
        (f"Where is the meeting on {topic}?", room),
        (f"What is the meeting about?", topic),
        (f"What does {n1} ask {n2} to bring?", "the latest figures"),
    ]
    question, answer = random.choice(qa_options)
    return f"{passage}\n{question}", answer


def _conv_project(n1, n2):
    project = rproject()
    deadline = random.choice(["next Friday", "end of the month", "March 15th", "two weeks from now", "the 22nd"])
    blocker = random.choice([
        "the vendor hasn't delivered the test samples",
        "we're still waiting for regulatory approval",
        "the simulation server has been down since Tuesday",
        "the client changed the specifications last minute",
        "two team members are out with the flu",
    ])
    solution = random.choice([
        "I've contacted their operations manager directly",
        "we could run the tests on the backup cluster",
        "I'll draft a revised scope document tonight",
        "we should request a one-week extension",
        "let's redistribute the workload across the remaining team",
    ])

    passage = f'{n1}: How is {project} progressing?\n{n2}: We hit a snag — {blocker}.\n{n1}: That could push us past the deadline. When is it due?\n{n2}: {deadline}. But {solution}.\n{n1}: Do that. Keep me posted.'

    qa_options = [
        (f"What is blocking {project}?", blocker),
        (f"When is {project} due?", deadline),
        (f"What solution does {n2} propose?", solution),
    ]
    question, answer = random.choice(qa_options)
    return f"{passage}\n{question}", answer


def _conv_travel(n1, n2):
    city = rloc()
    country = rcountry()
    airline = random.choice(["NorthStar Airlines", "Meridian Air", "CrossCurrent Airways", "Summit Aviation", "Coastal Express"])
    gate = random.choice(["Gate A3", "Gate B7", "Gate C12", "Gate D5", "Gate E9"])
    time = random.choice(["6:45 AM", "10:20 AM", "1:30 PM", "4:15 PM", "8:50 PM"])

    passage = f'{n1}: I booked the flight to {city}, {country} — it departs at {time} from {gate}.\n{n2}: Which airline?\n{n1}: {airline}. The return is three days later.\n{n2}: I\'ll arrange the hotel and ground transport.\n{n1}: Great. Make sure we have early check-in.'

    qa_options = [
        (f"What time does the flight depart?", time),
        (f"What gate is the flight to {city} leaving from?", gate),
        (f"Which airline is {n1} flying?", airline),
        (f"Where is {n1} traveling to?", f"{city}, {country}"),
    ]
    question, answer = random.choice(qa_options)
    return f"{passage}\n{question}", answer


def _conv_technical(n1, n2):
    device = rdevice()
    reading = random.choice(["0.47 volts", "3.2 millimeters", "18.6 degrees", "127 ohms", "42 psi", "0.003 seconds"])
    expected = random.choice(["within normal range", "slightly above threshold", "well below specification", "at the upper limit", "matching the baseline"])
    action = random.choice([
        "recalibrate using the secondary reference",
        "replace the gasket and retest",
        "log it and monitor for drift over the next 48 hours",
        "swap the sensor module with a spare from storage",
        "increase the sampling rate and take another measurement",
    ])

    passage = f'{n1}: I ran {device} through the morning test cycle.\n{n2}: What did you get?\n{n1}: The primary reading came in at {reading} — {expected}.\n{n2}: Any recommended follow-up?\n{n1}: Yes. We should {action}.'

    qa_options = [
        (f"What was the reading from {device}?", reading),
        (f"How does the reading compare to expectations?", expected),
        (f"What follow-up does {n1} recommend?", action),
    ]
    question, answer = random.choice(qa_options)
    return f"{passage}\n{question}", answer


def gen_multi_hop() -> Tuple[str, str]:
    """Multi-hop reasoning requiring combining multiple facts."""
    name1 = rname()
    name2 = rname()
    org = rorg()
    loc = rloc()
    country = rcountry()
    role = rrole()

    passages = [
        (f"{name1} works at {org}. {org} is headquartered in {loc}, {country}. {name2} is {name1}'s supervisor.",
         [
             (f"In which country does {name1} work?", country),
             (f"Where is {name1}'s workplace located?", f"{loc}, {country}"),
             (f"Who supervises {name1}?", name2),
         ]),
        (f"{name1} manages the {loc} branch of {org}. The branch reports to {name2}, who is the regional director based in {country}.",
         [
             (f"Who does the {loc} branch report to?", name2),
             (f"Where is {name2} based?", country),
             (f"Which branch does {name1} manage?", f"the {loc} branch"),
         ]),
        (f"{org} developed {rdevice()} for {rproject()}. {name1}, the lead {role}, presented the results to {name2} at the {loc} conference in {country}.",
         [
             (f"Who presented the results?", name1),
             (f"Where was the conference held?", f"{loc}"),
             (f"What is {name1}'s role?", f"lead {role}"),
         ]),
    ]
    passage, qa_options = random.choice(passages)
    question, answer = random.choice(qa_options)
    return f"{passage}\n{question}", answer


def gen_comparison() -> Tuple[str, str]:
    """Comparison between two entities."""
    name1 = rname()
    name2 = rname()
    org1 = rorg()
    org2 = pick(ORGANIZATIONS, exclude=org1)
    year1 = ryear()
    year2 = ryear()
    metric1 = random.choice(["$2.4 million", "$800,000", "$5 million", "$12 million", "$350,000"])
    metric2 = random.choice(["$1.8 million", "$600,000", "$3.2 million", "$9 million", "$450,000"])

    passages = [
        f"{org1}, led by {name1} since {year1}, secured funding of {metric1} for their latest initiative. Meanwhile, {org2} under {name2} received {metric2} in {year2}.",
        f"Two competing proposals emerged in the field. {name1} at {org1} obtained {metric1} in {year1}, while {name2} at {org2} was awarded {metric2} in {year2}.",
    ]
    passage = random.choice(passages)

    qa_options = [
        (f"How much funding did {org1} receive?", metric1),
        (f"Who leads {org2}?", name2),
        (f"When did {org2} receive its funding?", str(year2)),
        (f"How much was {name2}'s award?", metric2),
    ]
    question, answer = random.choice(qa_options)
    return f"{passage}\n{question}", answer


def gen_cause_effect() -> Tuple[str, str]:
    """Causal relationships."""
    event = random.choice([
        "a prolonged drought in the eastern provinces",
        "the failure of the main cooling system",
        "unexpected seismic activity along the coast",
        "a severe shortage of raw materials",
        "the discovery of contamination in the water supply",
        "a sudden drop in atmospheric pressure",
        "widespread network outages across the region",
    ])
    cause = random.choice([
        "an unusually warm winter that reduced snowpack by 60%",
        "a software update that introduced a critical timing error",
        "shifting tectonic plates beneath the continental shelf",
        "trade restrictions imposed by three neighboring nations",
        "runoff from an abandoned industrial site upstream",
        "rapid convective cooling from an approaching cold front",
        "a fiber optic cable severed during road construction",
    ])
    response = random.choice([
        "emergency water rationing was implemented across four districts",
        "engineers switched to the backup system within 90 minutes",
        "coastal evacuation orders were issued for 12,000 residents",
        "production was shifted to alternative synthetic substitutes",
        "three new filtration stations were deployed within a week",
        "all outdoor events were cancelled until conditions stabilized",
        "traffic was rerouted through satellite relay links",
    ])
    name = rname()
    loc = rloc()

    passage = f"In {loc}, {event} was traced to {cause}. According to {name}, the regional coordinator, {response}."

    qa_options = [
        (f"What caused {event.split(' in ')[0] if ' in ' in event else event}?", cause),
        (f"How did {loc} respond?", response),
        (f"Who is the regional coordinator in {loc}?", name),
    ]
    question, answer = random.choice(qa_options)
    return f"{passage}\n{question}", answer


# ==========================================================================
# Knowledge base generator (for memory pre-seeding)
# ==========================================================================

def gen_knowledge_fact() -> Tuple[str, str]:
    """
    Generate a single knowledge base fact.
    Returns (fact_text, entity_key) where entity_key identifies the main entity.
    """
    generators = [
        _kb_person, _kb_place, _kb_organization,
        _kb_material, _kb_device, _kb_event,
    ]
    return random.choice(generators)()


def _kb_person():
    name = rname()
    role = rrole()
    org = rorg()
    loc = rloc()
    country = rcountry()
    year = ryear()
    fact = f"{name} is a {role} at {org} in {loc}, {country}. {name} has been in this position since {year}."
    return fact, name


def _kb_place():
    loc = rloc()
    country = rcountry()
    pop = random.choice(["15,000", "42,000", "88,000", "130,000", "210,000"])
    feature = random.choice([
        "a series of underground thermal vents",
        "the oldest continuously operating lighthouse in the region",
        "a volcanic caldera lake",
        "extensive coral reef formations offshore",
        "a centuries-old canal network",
    ])
    fact = f"{loc} is a settlement in {country} with a population of {pop}. Its most notable feature is {feature}."
    return fact, loc


def _kb_organization():
    org = rorg()
    loc = rloc()
    country = rcountry()
    year = ryear()
    specialty = random.choice([
        "advanced materials research", "marine ecosystem monitoring",
        "satellite communication systems", "renewable energy storage",
        "computational drug discovery", "autonomous vehicle navigation",
        "seismic early warning systems", "quantum key distribution",
    ])
    fact = f"{org}, founded in {year} and headquartered in {loc}, {country}, specializes in {specialty}."
    return fact, org


def _kb_material():
    mat = rmaterial()
    prop = random.choice([
        "remains stable at temperatures exceeding 900°C",
        "is transparent to infrared radiation",
        "exhibits superconductivity below 40 kelvin",
        "absorbs vibrations across a wide frequency range",
        "self-repairs micro-fractures when exposed to UV light",
    ])
    app = random.choice([
        "next-generation heat shields", "optical sensor housings",
        "quantum computing substrates", "vibration-damping panels",
        "self-healing structural coatings",
    ])
    fact = f"The {mat} {prop}. It is primarily used in {app}."
    return fact, mat


def _kb_device():
    dev = rdevice()
    org = rorg()
    spec = random.choice([
        "measures atmospheric particulate matter down to 0.1 micrometers",
        "operates continuously for 18 months on a single charge",
        "processes acoustic signals in real time at 500 kHz",
        "detects structural fatigue in steel beams at a range of 50 meters",
        "maps subsurface geological formations to a depth of 200 meters",
    ])
    fact = f"{dev}, manufactured by {org}, {spec}."
    return fact, dev


def _kb_event():
    loc = rloc()
    country = rcountry()
    year = ryear()
    event_type = random.choice(["accord", "summit", "protocol", "convention", "compact"])
    outcome = random.choice([
        "standardized emission targets for participating nations",
        "created a multinational rapid-response research team",
        "established shared data repositories for climate monitoring",
        "introduced cross-border wildlife corridor protections",
        "set binding deadlines for transitioning to renewable energy",
    ])
    fact = f"The {loc} {event_type} of {year} was held in {country} and {outcome}."
    return fact, f"{loc} {event_type}"


# ==========================================================================
# QA from knowledge base (for pre-seeded memory training)
# ==========================================================================

def gen_kb_question(fact: str, entity: str) -> Tuple[str, str]:
    """
    Generate a question about a knowledge base fact.
    The question alone is the context (the answer should come from pre-seeded memory).
    """
    # Parse fact to extract answerable spans — use simple heuristic
    # The question only contains the entity name and a question word.
    # The answer requires reading the fact from memory.
    question_templates = [
        f"What do you know about {entity}?",
        f"Tell me about {entity}.",
        f"Describe {entity}.",
        f"What is {entity}?",
        f"Give me information on {entity}.",
    ]
    question = random.choice(question_templates)
    # Answer is a condensed form of the fact
    return question, fact


# ==========================================================================
# Build NOOP-target sequences
# ==========================================================================

def build_noop_sequences(pairs, tokenizer, seq_len, pad_id, bos_id, eos_id, noop_id):
    """Same as dataset._build_noop_sequences but standalone."""
    N = len(pairs)
    data = torch.full((N, seq_len), pad_id, dtype=torch.int32)
    targets = torch.full((N, seq_len), pad_id, dtype=torch.int32)
    actual_n = 0
    skipped = 0
    log_every = max(1, N // 20)

    for idx, (ctx_text, resp_text) in enumerate(pairs):
        ctx_ids = encode(tokenizer, ctx_text, max_len=seq_len - 4) if ctx_text else []
        max_resp = seq_len - 2 - len(ctx_ids)
        if max_resp < 2:
            skipped += 1
            continue
        resp_ids = encode(tokenizer, resp_text, max_len=max_resp) if resp_text else []
        if not resp_ids:
            skipped += 1
            continue

        row = [bos_id] + ctx_ids + resp_ids + [eos_id]
        n = len(row)
        if n > seq_len:
            row = row[:seq_len]
            n = seq_len
        data[actual_n, :n] = torch.tensor(row, dtype=torch.int32)

        tgt = [pad_id] * seq_len
        ctx_end = 1 + len(ctx_ids)
        for p in range(n - 1):
            if p < ctx_end:
                tgt[p] = noop_id
            else:
                tgt[p] = row[p + 1]
        targets[actual_n, :] = torch.tensor(tgt, dtype=torch.int32)
        actual_n += 1

        if idx % log_every == 0:
            print(f"    Building: {idx:,}/{N:,} ({100*idx/max(N,1):.0f}%)", flush=True)

    if skipped:
        print(f"    Skipped {skipped:,} sequences (too long or empty)")

    return data[:actual_n].clone(), targets[:actual_n].clone()


# ==========================================================================
# Main
# ==========================================================================

QA_GENERATORS = [
    gen_person_role,
    gen_project_details,
    gen_material_science,
    gen_geography,
    gen_device_specs,
    gen_publication,
    gen_event_history,
    gen_instruction,
    gen_conversation,
    gen_multi_hop,
    gen_comparison,
    gen_cause_effect,
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate Phase 5 synthetic training dataset")
    parser.add_argument("--data_dir", default="./data_cache")
    parser.add_argument("--total_qa", type=int, default=80_000,
                        help="Number of in-sequence QA pairs to generate")
    parser.add_argument("--total_kb", type=int, default=10_000,
                        help="Number of knowledge base facts for pre-seeding")
    parser.add_argument("--kb_qa_per_fact", type=int, default=2,
                        help="QA pairs per knowledge base fact (for memory-only training)")
    parser.add_argument("--tinystories_samples", type=int, default=50_000,
                        help="TinyStories replay examples (0 to skip)")
    parser.add_argument("--no_tinystories", action="store_true")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    tok_path = os.path.join(args.data_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print(f"✗ Tokenizer not found at {tok_path}")
        return
    tokenizer = load_tokenizer(tok_path)

    pad_id = tokenizer.token_to_id("<pad>") or 0
    bos_id = tokenizer.token_to_id("<bos>") or 2
    eos_id = tokenizer.token_to_id("<eos>") or 1
    noop_id = 6

    # ------------------------------------------------------------------
    # 1. Generate in-sequence QA pairs
    # ------------------------------------------------------------------
    print("=" * 64)
    print("  Generating in-sequence QA pairs")
    print("=" * 64)
    qa_pairs = []
    per_gen = args.total_qa // len(QA_GENERATORS)
    for gen_fn in QA_GENERATORS:
        for _ in range(per_gen):
            try:
                ctx, ans = gen_fn()
                qa_pairs.append((ctx, ans))
            except Exception:
                pass
    # Fill remainder
    while len(qa_pairs) < args.total_qa:
        gen_fn = random.choice(QA_GENERATORS)
        try:
            ctx, ans = gen_fn()
            qa_pairs.append((ctx, ans))
        except Exception:
            pass
    random.shuffle(qa_pairs)
    print(f"  Generated {len(qa_pairs):,} QA pairs across {len(QA_GENERATORS)} categories")

    # ------------------------------------------------------------------
    # 2. Generate knowledge base + memory-dependent QA
    # ------------------------------------------------------------------
    print()
    print("=" * 64)
    print("  Generating knowledge base for pre-seeding")
    print("=" * 64)
    kb_facts = []
    kb_qa_pairs = []
    seen_entities = set()
    attempts = 0
    while len(kb_facts) < args.total_kb and attempts < args.total_kb * 5:
        attempts += 1
        fact, entity = gen_knowledge_fact()
        if entity in seen_entities:
            continue
        seen_entities.add(entity)
        kb_facts.append({"fact": fact, "entity": entity})
        # Generate QA pairs for this fact
        for _ in range(args.kb_qa_per_fact):
            q, a = gen_kb_question(fact, entity)
            kb_qa_pairs.append((q, a))

    random.shuffle(kb_qa_pairs)
    print(f"  Generated {len(kb_facts):,} knowledge base facts")
    print(f"  Generated {len(kb_qa_pairs):,} memory-dependent QA pairs")

    # Save knowledge base to JSONL
    kb_path = os.path.join(args.data_dir, "knowledge_base.jsonl")
    with open(kb_path, "w") as f:
        for item in kb_facts:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved knowledge base → {kb_path}")

    # ------------------------------------------------------------------
    # 3. TinyStories replay
    # ------------------------------------------------------------------
    tiny_pairs = []
    if not args.no_tinystories and args.tinystories_samples > 0:
        print()
        print("=" * 64)
        print("  Loading TinyStories replay")
        print("=" * 64)
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        for item in ds:
            if len(tiny_pairs) >= args.tinystories_samples:
                break
            text = item["text"].strip()
            if text:
                tiny_pairs.append(("", text))
        print(f"  Loaded {len(tiny_pairs):,} stories")

    # ------------------------------------------------------------------
    # 4. Combine and build sequences
    # ------------------------------------------------------------------
    all_pairs = qa_pairs + kb_qa_pairs + tiny_pairs
    random.shuffle(all_pairs)

    n_qa = len(qa_pairs)
    n_kb = len(kb_qa_pairs)
    n_tiny = len(tiny_pairs)
    n_total = len(all_pairs)

    print()
    print("=" * 64)
    print("  Dataset Composition")
    print("=" * 64)
    print(f"  In-sequence QA:     {n_qa:>7,} ({100*n_qa/n_total:.1f}%)")
    print(f"  Memory-dep QA:      {n_kb:>7,} ({100*n_kb/n_total:.1f}%)")
    print(f"  TinyStories replay: {n_tiny:>7,} ({100*n_tiny/n_total:.1f}%)")
    print(f"  Total:              {n_total:>7,}")

    print()
    print("Tokenizing and building NOOP-target sequences...")
    data, targets = build_noop_sequences(
        all_pairs, tokenizer, args.seq_len,
        pad_id, bos_id, eos_id, noop_id,
    )

    # Save training cache
    cache_path = os.path.join(args.data_dir, "gen1_memory_qa_context_text_tokens_v3.pt")
    torch.save({"data": data, "targets": targets, "version": 3}, cache_path)

    # Stats
    noop_count = (targets == noop_id).sum().item()
    real_count = ((targets != pad_id) & (targets != noop_id)).sum().item()
    total_toks = noop_count + real_count
    tokens_per_epoch = data.shape[0] * args.seq_len

    print()
    print("=" * 64)
    print("  Phase 5 Dataset Ready")
    print("=" * 64)
    print(f"  Sequences:    {data.shape[0]:,}")
    print(f"  Shape:        {tuple(data.shape)}")
    print(f"  Cache:        {cache_path}")
    print(f"  KB facts:     {kb_path}")
    print(f"  Size:         {os.path.getsize(cache_path) / 1e6:.1f} MB")
    print(f"  NOOP tokens:  {noop_count:,} ({100*noop_count/max(total_toks,1):.1f}%)")
    print(f"  Real tokens:  {real_count:,} ({100*real_count/max(total_toks,1):.1f}%)")
    print(f"  Tokens/epoch: ~{tokens_per_epoch/1e6:.0f}M")
    print()
    print("  Training workflow:")
    print("    1. python preseed_memory.py          # pre-seed memory with KB facts")
    print("    2. python train_gen1.py               # train Phase 5a (frozen memory)")
    print("    3. python train_gen1.py --unfreeze     # train Phase 5b (unfrozen)")
    print("    4. python eval_gen1.py                 # evaluate memory recall")
    print("=" * 64)

    # Show samples
    print()
    print("  Sample entries:")
    for i in range(min(5, len(qa_pairs))):
        ctx, ans = qa_pairs[i]
        lines = ctx.split("\n")
        passage = lines[0][:100]
        question = lines[-1] if len(lines) > 1 else "?"
        print(f"    [{i}] {passage}...")
        print(f"        Q: {question}")
        print(f"        A: {ans}")
        print()


if __name__ == "__main__":
    main()
