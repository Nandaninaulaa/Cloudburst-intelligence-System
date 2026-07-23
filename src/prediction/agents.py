import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =====================================================
# WEATHER MONITORING AGENT
# =====================================================

class WeatherMonitoringAgent:

    def run(self, weather):

        severity = "Normal"

        if weather["rainfall"] > 100:
            severity = "Severe"

        elif weather["rainfall"] > 50:
            severity = "Moderate"

        summary = (
            f"Temperature {weather['temperature']}°C, "
            f"Humidity {weather['humidity']}%, "
            f"Pressure {weather['pressure']} hPa, "
            f"Rainfall {weather['rainfall']} mm, "
            f"Wind Speed {weather['wind_speed']} m/s."
        )

        recommendation = (
            "Monitor weather conditions continuously."
            if severity == "Normal"
            else "Abnormal weather detected. Increased monitoring required."
        )

        return {
            "weather_summary": summary,
            "weather_severity": severity,
            "recommendation": recommendation
        }


# =====================================================
# RISK ASSESSMENT AGENT
# =====================================================

class RiskAssessmentAgent:

    def run(self, probability, confidence):

        if probability < 30:
            risk = "Low"

        elif probability < 60:
            risk = "Medium"

        elif probability < 80:
            risk = "High"

        else:
            risk = "Extreme"

        priority_score = round((probability * confidence) / 100, 2)

        return {
            "risk_level": risk,
            "confidence": confidence,
            "reason":
                f"Cloudburst probability is {probability}% "
                f"with confidence {confidence}%",
            "priority_score": priority_score
        }


# =====================================================
# EVACUATION AGENT
# =====================================================

class EvacuationPlanningAgent:

    def run(self, risk_level):

        plans = {
            "Low": {
                "evacuation_required": False,
                "urgency": "Low",
                "action_plan": [
                    "Continue monitoring weather updates"
                ]
            },

            "Medium": {
                "evacuation_required": False,
                "urgency": "Medium",
                "action_plan": [
                    "Prepare emergency kits",
                    "Stay alert"
                ]
            },

            "High": {
                "evacuation_required": True,
                "urgency": "High",
                "action_plan": [
                    "Alert residents",
                    "Prepare evacuation routes",
                    "Move vulnerable groups"
                ]
            },

            "Extreme": {
                "evacuation_required": True,
                "urgency": "Immediate",
                "action_plan": [
                    "Immediate evacuation",
                    "Activate emergency shelters",
                    "Deploy rescue teams"
                ]
            }
        }

        return plans[risk_level]


# =====================================================
# RESOURCE AGENT
# =====================================================

class ResourceAllocationAgent:

    def run(self, risk_level):

        mapping = {

            "Low": {
                "rescue_teams": 1,
                "medical_teams": 1,
                "vehicles": 1,
                "resource_status": "Minimal resources required"
            },

            "Medium": {
                "rescue_teams": 3,
                "medical_teams": 2,
                "vehicles": 3,
                "resource_status": "Standby deployment"
            },

            "High": {
                "rescue_teams": 6,
                "medical_teams": 4,
                "vehicles": 8,
                "resource_status": "Deploy response units"
            },

            "Extreme": {
                "rescue_teams": 10,
                "medical_teams": 8,
                "vehicles": 15,
                "resource_status": "Full emergency deployment"
            }
        }

        return mapping[risk_level]


# =====================================================
# COMMUNICATION AGENT
# =====================================================

class CommunicationAgent:

    def run(self, location, risk_level):

        message = (
            f"⚠ Cloudburst Warning: {risk_level} risk "
            f"detected in {location}."
        )

        if risk_level == "Extreme":
            message += " Immediate evacuation advised."

        return {
            "sms_alert": message,
            "email_alert": message,
            "dashboard_alert": message
        }


# =====================================================
# MASTER COORDINATOR
# =====================================================

class CoordinatorAgent:

    def __init__(self):

        self.weather_agent = WeatherMonitoringAgent()
        self.risk_agent = RiskAssessmentAgent()
        self.evac_agent = EvacuationPlanningAgent()
        self.resource_agent = ResourceAllocationAgent()
        self.comm_agent = CommunicationAgent()

    def execute(self,
                location,
                latitude,
                longitude,
                temperature,
                humidity,
                pressure,
                wind_speed,
                rainfall,
                probability,
                confidence):

        weather_data = {
            "temperature": temperature,
            "humidity": humidity,
            "pressure": pressure,
            "wind_speed": wind_speed,
            "rainfall": rainfall
        }

        weather_result = self.weather_agent.run(weather_data)

        risk_result = self.risk_agent.run(
            probability,
            confidence
        )

        evac_result = self.evac_agent.run(
            risk_result["risk_level"]
        )

        resource_result = self.resource_agent.run(
            risk_result["risk_level"]
        )

        comm_result = self.comm_agent.run(
            location,
            risk_result["risk_level"]
        )

        final_summary = (
            f"{risk_result['risk_level']} risk detected "
            f"for {location}. "
            f"Priority Score: {risk_result['priority_score']}"
        )

        return {

            "weather_agent": weather_result,

            "risk_agent": risk_result,

            "evacuation_agent": evac_result,

            "resource_agent": resource_result,

            "communication_agent": comm_result,

            "final_summary": final_summary
        }