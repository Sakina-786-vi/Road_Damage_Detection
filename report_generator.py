"""
report_generator.py - LangChain + Google Gemini powered road inspection report generator.

Generates professional reports, repair recommendations, risk assessments,
and suggested repair timelines from damage analysis results.
"""

import os
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser


PRIORITY_TIMELINES = {
    "Low Priority": "12–24 months (routine maintenance cycle)",
    "Medium Priority": "6–12 months (next scheduled maintenance window)",
    "High Priority": "1–3 months (urgent intervention required)",
    "Critical Priority": "Immediate action required (within 1–2 weeks)",
}

COST_RATES = {
    # INR per square meter for localized patch repair
    "Low Priority": 850,
    "Medium Priority": 1250,
    "High Priority": 1750,
    "Critical Priority": 2400,
}


def estimate_repair_cost(
    damage_percentage: float,
    priority: str,
    road_area_m2: float = 120,
    num_damaged_regions: int = 1,
) -> dict:
    """
    Estimate repair cost for a localized pothole / patch repair.

    Args:
        damage_percentage: Percentage of road surface damaged (0-100)
        priority: One of the four priority strings
        road_area_m2: Assumed road section area in square meters (default 120 m²)
        num_damaged_regions: Detected damaged regions to scale the repair effort

    Returns:
        dict with 'damaged_area_m2', 'cost_per_m2', 'estimated_cost', 'currency'
    """
    rate = COST_RATES.get(priority, 250)
    detected_regions = max(int(num_damaged_regions), 1)

    # Convert the image-level damage score into a repairable patch size.
    # This keeps a single pothole from being priced like a full resurfacing job.
    damage_area_estimate = (damage_percentage / 100) * road_area_m2
    patch_area_estimate = max(0.35 * detected_regions, 0.5)
    damaged_area = min(damage_area_estimate, patch_area_estimate, 4.0)

    # Small repair jobs have a call-out charge, but we keep it modest.
    mobilisation_cost = 650 + (150 * max(detected_regions - 1, 0))

    # Apply a gentle priority multiplier instead of a large markup.
    priority_multiplier = {
        "Low Priority": 1.00,
        "Medium Priority": 1.10,
        "High Priority": 1.20,
        "Critical Priority": 1.35,
    }.get(priority, 1.10)

    base_cost = damaged_area * rate
    total_cost = (mobilisation_cost + base_cost) * priority_multiplier

    return {
        "damaged_area_m2": round(damaged_area, 2),
        "cost_per_m2": rate,
        "estimated_cost": round(total_cost, 2),
        "currency": "INR",
    }


class ReportGenerator:
    """Generates AI-powered road inspection reports using LangChain + Gemini."""

    def __init__(self, gemini_api_key: str | None = None):
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Gemini API key not set. "
                "Set the GEMINI_API_KEY environment variable or pass it directly."
            )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a senior civil engineer specialising in road infrastructure assessment. "
             "Generate professional, concise, and actionable inspection reports. "
             "Use clear section headings and bullet points where appropriate. "
             "Be specific with numbers and technical terminology."
             ),
            ("human",
             """Generate a professional road damage inspection report based on the following analysis data:

=== DAMAGE METRICS ===
• Damage Area: {damage_percentage}% of road surface
• Damaged Regions Detected: {num_damaged_regions}
• Crack Density Index: {crack_density}
• Texture Roughness Score: {texture_roughness} / 1.0
• Dark Surface Percentage: {dark_surface_percentage}%

=== ASSESSMENT RESULTS ===
• Priority Classification: {priority}
• Model Confidence: {confidence}%
• Repair Timeline: {timeline}

=== COST ESTIMATE ===
• Estimated Damaged Area: {damaged_area_m2} m²
• Repair Cost Estimate: ₹{estimated_cost} INR
• Cost per m²: ₹{cost_per_m2}

Please provide a structured report including:
1. **Executive Summary** (2-3 sentences summarising the condition)
2. **Detailed Damage Assessment** (technical analysis of each metric)
3. **Risk Assessment** (safety, structural, traffic impact risks)
4. **Repair Recommendations** (specific methods and materials)
5. **Suggested Repair Timeline** (phased schedule with milestones)
6. **Inspector Notes** (any caveats, further investigations needed)

Format the output in clean Markdown.
"""
             ),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(self, analysis: dict, priority_result: dict, cost_result: dict) -> str:
        """
        Generate the full inspection report.

        Args:
            analysis: Output from RoadDamageAnalyzer.extract_features()
            priority_result: Output from DamagePriorityModel.predict()
            cost_result: Output from estimate_repair_cost()

        Returns:
            Markdown-formatted report string
        """
        timeline = PRIORITY_TIMELINES.get(priority_result["priority"], "To be determined")

        return self.chain.invoke({
            "damage_percentage": analysis["damage_percentage"],
            "num_damaged_regions": analysis["num_damaged_regions"],
            "crack_density": analysis["crack_density"],
            "texture_roughness": analysis["texture_roughness"],
            "dark_surface_percentage": analysis["dark_surface_percentage"],
            "priority": priority_result["priority"],
            "confidence": priority_result["confidence"],
            "timeline": timeline,
            "damaged_area_m2": cost_result["damaged_area_m2"],
            "estimated_cost": f"{cost_result['estimated_cost']:,.2f}",
            "cost_per_m2": cost_result["cost_per_m2"],
        })
