import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

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
    temperature=0.7,
    max_tokens=1000,
)

agent = create_agent(
    model=model,
    system_prompt="Vous êtes un assistant IA utile qui fournit des réponses claires et concises."
)

# Test du cours
print("Test agent...")
reponse = agent.invoke({
    "messages": [HumanMessage(content="quel est le capital de madagascar ?")]
})
print("\nRéponse :\n", reponse["messages"][-1].content)