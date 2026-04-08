from json import tool
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.tools import tool

# Charger .env (RACINE projet)
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("ERREUR: Clé manquante dans .env")
    exit(1)

print("Clé chargée")

model = ChatOpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini",
    temperature=0.1,
    max_tokens=1000,
)

@tool
def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert temperature between Celsius and Fahrenheit.
    
    Args:
        value: The temperature value to convert
        from_unit: Source unit ("celsius" or "fahrenheit")
        to_unit: Target unit ("celsius" or "fahrenheit")
    
    Returns:
        Converted temperature value
    """
    if from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
        return (value * 9/5) + 32
    elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
        return (value - 32) * 5/9
    elif from_unit.lower() == to_unit.lower():
        return value
    else:
        raise ValueError(f"Unsupported conversion from {from_unit} to {to_unit}")


agent = create_agent(
    model=model,
    system_prompt="You are a helpful assistant that can convert temperatures between Celsius and Fahrenheit. Always use the convert_temperature tool when users ask for temperature conversions.",
    tools=[convert_temperature]
)

# Test du cours
print("Test agent...")
tests = [
    "Convertis 25°C en Fahrenheit",
    "Quelle est 98.6°F en Celsius ?",
    "Quel est le capital de Madagascar ?"
]

for test in tests:
    print(f"\n--- Test: {test} ---")
    reponse = agent.invoke({"messages": [HumanMessage(content=test)]})
    print("Agent:", reponse["messages"][-1].content)

print("=== Test 1: Conversion simple ===")
response = agent.invoke({"messages": [HumanMessage("Convertis 25 degrés Celsius en Fahrenheit")]})
print(response["messages"][-1].content)

print("\n=== Test 2: Conversion inverse ===")
response = agent.invoke({"messages": [HumanMessage("Que sont 77 degrés Fahrenheit en Celsius ?")]})
print(response["messages"][-1].content)

print("\n=== Test 3: Question complexe ===")
response = agent.invoke({"messages": [HumanMessage("S'il fait 20°C dehors, ça fait combien en Fahrenheit ? Est-ce chaud ou froid ?")]})
print(response["messages"][-1].content)