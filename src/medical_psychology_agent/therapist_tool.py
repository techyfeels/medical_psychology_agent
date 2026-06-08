"""
Therapist & psychiatrist directory for Virginia / Washington DC area.
Data sourced from public directories: Psychology Today, clinic websites (2024-2025).
"""

from __future__ import annotations

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Directory data
# ---------------------------------------------------------------------------

CLINICS: List[Dict] = [
    {
        "id": "clinic_001",
        "name": "Shugarman Psychiatric & Counseling",
        "type": "Private Practice",
        "description": (
            "Board-certified psychiatrists and licensed therapists serving adults, "
            "adolescents, couples, and families in Northern Virginia and Greater DC since 2010. "
            "Also offers forensic psychiatric evaluations."
        ),
        "specialties": [
            "Anxiety", "Depression", "PTSD", "Trauma", "OCD",
            "Bipolar Disorder", "ADHD", "Couples Therapy", "Forensic Psychiatry",
        ],
        "therapies": [
            "Medication Management", "Psychotherapy",
            "Cognitive Behavioral Therapy (CBT)", "Teletherapy",
        ],
        "location": {
            "address": "901 N Washington St, Suite 601",
            "city": "Alexandria",
            "state": "VA",
            "zip": "22314",
            "area": "Alexandria / Northern Virginia",
        },
        "contact": {
            "phone": "(703) 596-1024",
            "fax": "(703) 596-1573",
            "email": None,
            "website": "https://shugarmanpsychiatric.com",
            "contact_form": "https://shugarmanpsychiatric.com/contact-us/",
        },
        "providers": [
            {
                "name": "Dr. Ryan Shugarman, MD",
                "title": "Board-Certified Adult & Forensic Psychiatrist",
                "profile_url": "https://shugarmanpsychiatric.com/ryan-shugarman-md/",
            },
            {
                "name": "Dr. James Rives, MD",
                "title": "Board-Certified Psychiatrist — specializes in U.S. Military Veterans",
                "profile_url": "https://shugarmanpsychiatric.com/james-rives-md/",
            },
            {
                "name": "Kristin Harkins, LMFT",
                "title": "Licensed Marriage & Family Therapist",
                "profile_url": None,
            },
            {
                "name": "Shilpa Krishnan, Ph.D.",
                "title": "Forensic Psychologist",
                "profile_url": None,
            },
        ],
        "ages_served": "Adolescents (13+) and Adults",
        "session_types": ["In-person", "Telehealth"],
        "accepting_patients": True,
        "insurance": ["Contact clinic for accepted plans"],
        "languages": ["English"],
        "source": "https://shugarmanpsychiatric.com",
    },
    {
        "id": "clinic_002",
        "name": "Genesis Psychiatric Solutions",
        "type": "Psychiatry Practice",
        "description": (
            "Board-certified psychiatry with an integrative, holistic, patient-centered approach. "
            "Treats anxiety, depression, ADHD, and other mental health conditions. "
            "Telepsychiatry available."
        ),
        "specialties": [
            "Anxiety", "Depression", "ADHD", "Bipolar Disorder",
            "Insomnia", "PTSD", "Life Transitions",
        ],
        "therapies": [
            "Medication Management", "Psychotherapy", "Telepsychiatry",
        ],
        "locations": [
            {
                "name": "Fairfax Office",
                "address": "10339 Democracy Lane, Suite A",
                "city": "Fairfax",
                "state": "VA",
                "zip": "22030",
                "phone": "(571) 748-4971",
            },
            {
                "name": "Washington DC Office",
                "address": "419 7th St. NW, Suite 405",
                "city": "Washington",
                "state": "DC",
                "zip": "20004",
                "phone": "(202) 410-2381",
            },
            {
                "name": "Alexandria Office",
                "address": "901 N. Washington St, Suite 204",
                "city": "Alexandria",
                "state": "VA",
                "zip": "22314",
                "phone": "(571) 384-3341",
            },
        ],
        "contact": {
            "phone": "(571) 748-4971",
            "email": None,
            "website": "https://www.genesispsychiatricsolutions.com",
            "booking": "https://www.genesispsychiatricsolutions.com/schedule",
            "contact_form": "https://www.genesispsychiatricsolutions.com/contactus",
        },
        "providers": [
            {
                "name": "Dr. Ifeanyi Olele, MD",
                "title": "Board-Certified Psychiatrist",
                "profile_url": "https://www.genesispsychiatricsolutions.com/contents/about/meet-dr-olele",
            },
        ],
        "ages_served": "Adolescents (14+) and Adults",
        "session_types": ["In-person", "Telehealth / Telepsychiatry"],
        "accepting_patients": True,
        "insurance": ["Most major insurance — contact for details"],
        "languages": ["English"],
        "source": "https://www.genesispsychiatricsolutions.com",
    },
    {
        "id": "clinic_003",
        "name": "Columbia Mental Health",
        "type": "Multi-location Mental Health Clinic",
        "description": (
            "Full-spectrum mental health services — therapy and psychiatry — across DC, Maryland, "
            "and Virginia. Accepts most major insurance including Medicaid and Medicare. "
            "Team includes psychiatrists, nurse practitioners, therapists, and social workers."
        ),
        "specialties": [
            "Anxiety", "Depression", "PTSD", "Trauma", "Grief",
            "Anger Management", "CBT", "DBT", "EMDR",
            "Child & Adolescent Mental Health",
        ],
        "therapies": [
            "Cognitive Behavioral Therapy (CBT)",
            "Dialectical Behavior Therapy (DBT)",
            "EMDR",
            "Medication Management",
            "Telehealth Therapy",
        ],
        "locations": [
            {"city": "Alexandria", "state": "VA", "area": "Alexandria, VA", "phone": "(703) 682-8263"},
            {"city": "Arlington", "state": "VA", "area": "Arlington, VA", "phone": "(703) 682-8263"},
            {"city": "Washington", "state": "DC", "area": "Washington DC", "phone": "(703) 682-8263"},
        ],
        "contact": {
            "phone": "(703) 682-8263",
            "email": None,
            "website": "https://www.columbiapsychiatry-dc.com",
            "locations_page": "https://www.columbiapsychiatry-dc.com/locations/",
            "team_page": "https://www.columbiapsychiatry-dc.com/about-us/meet-our-team/",
        },
        "providers": [],
        "ages_served": "Children, Adolescents, Adults",
        "session_types": ["In-person", "Telehealth"],
        "accepting_patients": True,
        "insurance": ["Most major insurance", "Medicaid", "Medicare"],
        "languages": ["English", "Multiple languages — varies by provider"],
        "source": "https://www.columbiapsychiatry-dc.com",
    },
    {
        "id": "clinic_004",
        "name": "Inova Behavioral Health Services",
        "type": "Hospital-based Mental Health System",
        "description": (
            "Full range of behavioral health services for Northern Virginia and DC. "
            "Includes outpatient therapy, psychiatry, crisis walk-in (IPAC), and inpatient care. "
            "Serves all ages — children, adolescents, and adults."
        ),
        "specialties": [
            "Anxiety", "Depression", "PTSD", "Bipolar Disorder",
            "Schizophrenia", "Addiction", "Crisis Intervention",
            "Child & Adolescent Mental Health", "OCD",
        ],
        "therapies": [
            "Medication Management", "Individual Therapy",
            "Group Therapy", "Crisis Services", "Telehealth",
        ],
        "locations": [
            {
                "name": "IPAC — Urgent Psychiatric Care (Walk-in)",
                "address": "8221 Willow Oaks Corporate Dr, Suite 4-420",
                "city": "Fairfax",
                "state": "VA",
                "zip": "22031",
                "phone": "(571) 623-3515",
                "note": "Walk-in or call · Mon–Fri 8am–4pm",
            },
            {
                "name": "Merrifield Center",
                "city": "Fairfax",
                "state": "VA",
                "phone": "(703) 289-7599",
            },
            {
                "name": "Executive Park",
                "city": "Fairfax",
                "state": "VA",
                "phone": "(703) 852-7020",
            },
            {
                "name": "Mount Vernon (serves Alexandria area)",
                "city": "Alexandria area",
                "state": "VA",
                "phone": "(703) 660-8100",
            },
            {
                "name": "Loudoun County",
                "city": "Leesburg",
                "state": "VA",
                "phone": "(703) 737-2110",
            },
        ],
        "contact": {
            "phone": "(571) 623-3500",
            "email": None,
            "website": "https://www.inova.org/our-services/inova-behavioral-health-services",
            "access_page": "https://www.inova.org/our-services/inova-behavioral-health-services/access-to-care",
        },
        "providers": [],
        "ages_served": "All ages (children, adolescents, adults)",
        "session_types": ["In-person", "Telehealth", "Urgent walk-in (IPAC)"],
        "accepting_patients": True,
        "insurance": ["Most major insurance", "Medicaid", "Medicare"],
        "languages": ["English", "Multiple languages — varies by location"],
        "source": "https://www.inova.org",
    },
    {
        "id": "clinic_005",
        "name": "VHC Health — Behavioral & Mental Health",
        "type": "Hospital-based Psychiatry Unit",
        "description": (
            "The only hospital-based psychiatric unit in Arlington. Provides outpatient, "
            "inpatient, and addiction services. Includes 24/7 online scheduling and "
            "same-hour virtual urgent care."
        ),
        "specialties": [
            "Anxiety", "Depression", "PTSD", "Bipolar Disorder",
            "Schizophrenia", "Addiction", "Adolescent Mental Health",
        ],
        "therapies": [
            "Psychiatry", "Psychotherapy", "Medication Management",
            "Addiction Treatment", "Telehealth",
        ],
        "locations": [
            {
                "name": "Outpatient Behavioral Health",
                "address": "1715 N. George Mason Drive, Suite 201",
                "city": "Arlington",
                "state": "VA",
                "zip": "22205",
                "phone": "(703) 558-6750",
            },
            {
                "name": "Psychiatric & Addiction Services",
                "address": "1701 N. George Mason Drive, Behavioral Health Unit",
                "city": "Arlington",
                "state": "VA",
                "zip": "22205",
                "phone": "(703) 558-6750",
            },
        ],
        "contact": {
            "phone": "(703) 558-6750",
            "email": None,
            "website": "https://www.vhchealth.org/medical-services/behavioral-mental-health/",
            "scheduling": "https://www.vhchealth.org",
        },
        "providers": [],
        "ages_served": "Adolescents and Adults",
        "session_types": ["In-person", "Telehealth", "24/7 Online Scheduling"],
        "accepting_patients": True,
        "insurance": ["Most major insurance — contact for details"],
        "languages": ["English"],
        "source": "https://www.vhchealth.org",
    },
]

INDIVIDUAL_PROVIDERS: List[Dict] = [
    {
        "id": "ind_001",
        "name": "Dr. Joanna Chango-James",
        "title": "Licensed Clinical Psychologist",
        "type": "Individual Therapist",
        "description": "Specializes in OCD, panic disorder, social anxiety, PTSD, and insomnia. Uses evidence-based CBT and exposure therapy.",
        "specialties": ["Anxiety", "OCD", "Panic Disorder", "Social Anxiety", "PTSD", "Insomnia", "Chronic Pain"],
        "therapies": ["Cognitive Behavioral Therapy (CBT)", "Exposure & Response Prevention (ERP)"],
        "location": {"city": "Arlington", "state": "VA", "area": "Arlington, VA"},
        "contact": {
            "phone": None,
            "email": None,
            "website": None,
            "directory_profile": "https://www.psychologytoday.com/us/therapists/va/arlington?category=anxiety",
        },
        "ages_served": "Adults",
        "session_types": ["In-person", "Telehealth"],
        "accepting_patients": True,
        "languages": ["English"],
        "source": "Psychology Today",
    },
    {
        "id": "ind_002",
        "name": "Chava Nerenberg",
        "title": "Licensed Clinical Social Worker (LCSW)",
        "type": "Individual Therapist",
        "description": "Specializes in complex PTSD and developmental trauma. Uses EMDR and Cognitive Processing Therapy (CPT).",
        "specialties": ["Complex PTSD", "Trauma", "Developmental Trauma", "Anxiety", "Depression"],
        "therapies": ["EMDR", "Cognitive Processing Therapy (CPT)", "Trauma-Focused Therapy"],
        "location": {"city": "Arlington", "state": "VA", "area": "Arlington, VA"},
        "contact": {
            "phone": None,
            "email": None,
            "website": None,
            "directory_profile": "https://www.psychologytoday.com/us/therapists/va/arlington?category=trauma-and-ptsd",
        },
        "ages_served": "Adults",
        "session_types": ["In-person", "Telehealth"],
        "accepting_patients": True,
        "languages": ["English"],
        "source": "Psychology Today",
    },
    {
        "id": "ind_003",
        "name": "Josefina (LCSW)",
        "title": "Licensed Clinical Social Worker",
        "type": "Individual Therapist",
        "description": "Provides specialized care for anxiety, PTSD, and depression using CBT, DBT, and mindfulness-based interventions.",
        "specialties": ["Anxiety", "PTSD", "Depression", "Mindfulness"],
        "therapies": ["CBT", "DBT", "Mindfulness-Based Therapy"],
        "location": {"city": "Arlington", "state": "VA", "area": "Arlington, VA"},
        "contact": {
            "phone": None,
            "email": None,
            "website": None,
            "directory_profile": "https://www.psychologytoday.com/us/therapists/va/arlington?category=anxiety",
        },
        "ages_served": "Adults",
        "session_types": ["In-person", "Telehealth"],
        "accepting_patients": True,
        "languages": ["English", "Spanish"],
        "source": "Psychology Today",
    },
    {
        "id": "ind_004",
        "name": "Dr. Lisa Kruger",
        "title": "Licensed Clinical Psychologist",
        "type": "Individual Therapist",
        "description": "20+ years of experience helping clients work through anxiety, depression, and the impact of past trauma.",
        "specialties": ["Anxiety", "Depression", "Trauma", "PTSD", "Life Transitions"],
        "therapies": ["CBT", "Trauma-Focused Therapy", "Psychodynamic Therapy"],
        "location": {"city": "Alexandria", "state": "VA", "area": "Alexandria, VA"},
        "contact": {
            "phone": None,
            "email": None,
            "website": None,
            "directory_profile": "https://www.psychologytoday.com/us/therapists/va/alexandria?category=anxiety",
        },
        "ages_served": "Adults",
        "session_types": ["In-person", "Telehealth"],
        "accepting_patients": True,
        "languages": ["English"],
        "source": "Psychology Today",
    },
]

ALL_PROVIDERS = CLINICS + INDIVIDUAL_PROVIDERS

# ---------------------------------------------------------------------------
# Search logic
# ---------------------------------------------------------------------------

SPECIALTY_MAP: Dict[str, List[str]] = {
    "anxiety":       ["Anxiety", "OCD", "Panic Disorder", "Social Anxiety"],
    "depression":    ["Depression", "Mood Disorder"],
    "ptsd":          ["PTSD", "Trauma", "Complex PTSD"],
    "trauma":        ["Trauma", "PTSD", "Complex PTSD", "Developmental Trauma"],
    "adhd":          ["ADHD"],
    "bipolar":       ["Bipolar Disorder"],
    "ocd":           ["OCD"],
    "addiction":     ["Addiction"],
    "eating":        ["Eating Disorder"],
    "insomnia":      ["Insomnia"],
    "couples":       ["Couples Therapy"],
    "children":      ["Child & Adolescent Mental Health"],
    "adolescent":    ["Child & Adolescent Mental Health", "Adolescent Mental Health"],
    "schizophrenia": ["Schizophrenia"],
    "crisis":        ["Crisis Intervention"],
}

AREA_MAP: Dict[str, List[str]] = {
    "dc":           ["Washington", "DC"],
    "washington":   ["Washington", "DC"],
    "arlington":    ["Arlington"],
    "alexandria":   ["Alexandria"],
    "fairfax":      ["Fairfax"],
    "virginia":     ["VA", "Alexandria", "Arlington", "Fairfax"],
    "northern virginia": ["Alexandria", "Arlington", "Fairfax"],
}


class TherapistFinder:
    """Search the therapist directory by specialty and/or location area."""

    def __init__(self) -> None:
        self.directory = ALL_PROVIDERS

    def search(
        self,
        specialty: Optional[str] = None,
        area: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict]:
        results = self.directory

        if specialty:
            kw = specialty.lower()
            # Expand using map if available
            target_specs = None
            for key, specs in SPECIALTY_MAP.items():
                if key in kw or kw in key:
                    target_specs = [s.lower() for s in specs]
                    break

            if target_specs:
                results = [
                    p for p in results
                    if any(t in s.lower() for s in p.get("specialties", []) for t in target_specs)
                ]
            else:
                results = [
                    p for p in results
                    if any(kw in s.lower() for s in p.get("specialties", []))
                ]

        if area:
            area_lower = area.lower()
            target_areas = None
            for key, areas in AREA_MAP.items():
                if key in area_lower or area_lower in key:
                    target_areas = [a.lower() for a in areas]
                    break

            def _location_matches(provider: Dict) -> bool:
                loc = provider.get("location", {})
                locs = provider.get("locations", [])
                all_locs = ([loc] if loc else []) + (locs if isinstance(locs, list) else [])
                loc_str = " ".join(
                    f"{l.get('city','')} {l.get('state','')} {l.get('area','')}"
                    for l in all_locs
                ).lower()
                if target_areas:
                    return any(a in loc_str for a in target_areas)
                return area_lower in loc_str

            results = [p for p in results if _location_matches(p)]

        return results[:limit]

    def get_all(self, limit: int = 10) -> List[Dict]:
        return self.directory[:limit]

    def format_for_agent(self, providers: List[Dict]) -> str:
        """Compact text summary for use in agent context."""
        if not providers:
            return "No providers found in the directory for the given criteria."

        lines = ["=== PROVIDER DIRECTORY (Virginia / Washington DC) ==="]
        for p in providers:
            name = p["name"]
            ptype = p.get("type", "Provider")
            specs = ", ".join(p.get("specialties", [])[:5])
            phone = p.get("contact", {}).get("phone", "See website")
            website = p.get("contact", {}).get("website") or p.get("contact", {}).get("directory_profile", "")
            loc = p.get("location") or (p.get("locations") or [{}])[0]
            city = loc.get("city", "")
            state = loc.get("state", "")

            lines.append(
                f"\n• {name} | {ptype}\n"
                f"  Location: {city}, {state}\n"
                f"  Specialties: {specs}\n"
                f"  Phone: {phone}\n"
                f"  Website: {website}"
            )
        return "\n".join(lines)


_finder_instance: Optional[TherapistFinder] = None


def get_finder() -> TherapistFinder:
    global _finder_instance
    if _finder_instance is None:
        _finder_instance = TherapistFinder()
    return _finder_instance
