import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage

# Load environment variables from .env file
load_dotenv()

# Configure OpenRouter model
model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000,
)
# Créer un agent spécialisé en planification logistique 
logistics_agent = create_agent(
    model=model,
    system_prompt="""Vous êtes un expert en logistique de voyage. Vous gérez la planification pratique des voyages : 
    - Calculer les distances entre les lieux et les temps de trajet 
    - Estimer les coûts de transport, d'hébergement et d'activités 
    - Optimiser les itinéraires et suggérer des parcours efficaces 
    - Prendre en compte les fuseaux horaires, la météo et les contraintes pratiques 
    . Fournir systématiquement des informations logistiques concises, claires et pratiques.""" 
)
# Créer un agent spécialisé en recommandations 
recommendations_agent = create_agent ( 
    model=model, 
    system_prompt="""Vous êtes un spécialiste des recommandations de voyage. Vous suggérez des expériences et des activités : 
    - Recommander les principales attractions, les monuments et les lieux incontournables 
    - Suggérer des restaurants, la cuisine locale et des expériences culinaires 
    - Recommander des activités culturelles, des événements et des expériences locales 
    - Fournir des informations sur les coutumes locales, les meilleures périodes pour visiter et les trésors cachés. 
    Fournir systématiquement des recommandations brèves, attrayantes et personnalisées.""" 
)
# Agent Météo
weather_agent = create_agent(
    model=model,
    system_prompt="Expert météo voyages. Fournissez prévisions 3-7 jours, impact sur activités, conseils tenues."
)

# Agent Réservation  
booking_agent = create_agent(
    model=model,
    system_prompt="Expert réservations. Vérifiez disponibilités vols/hôtels, comparez prix, timing optimal."
)

# Agent Budget
budget_agent = create_agent(
    model=model,
    system_prompt="Optimiseur budget. Analysez dépenses, trouvez économies, priorisez, suivez budget restant."
)
@tool
def plan_logistics_agent(trip_request: str) -> str:
    """
    Planifie la logistique de voyage : distances, temps, coûts et itinéraires.
    Utilisez cet outil pour calculer les informations pratiques de voyage et optimiser les itinéraires.
    
    Args:
        trip_request: Détails du voyage (ex: "3 jours à Paris, budget 1500€, depuis Londres")
    
    Retourne:
        Informations logistiques : distances, temps de trajet, coûts et suggestions d'itinéraires
    """
    response = logistics_agent.invoke({"messages": [HumanMessage(f"Planifie la logistique pour ce voyage : {trip_request}")]})
    return response["messages"][-1].content
@tool
def get_recommendations_agent(trip_details: str) -> str:
    """
    Obtenez des recommandations de voyage pour attractions, restaurants et activités.
    Utilisez cet outil pour suggérer quoi voir, faire et manger à la destination.
    
    Args:
        trip_details: Destination et infos voyage (ex: "3 jours à Paris, intéressé par l'art et la nourriture")
    
    Retourne:
        Recommandations : attractions, restaurants, activités et insights culturels
    """
    response = recommendations_agent.invoke({"messages": [HumanMessage(f"Fournis des recommandations pour : {trip_details}")]})
    return response["messages"][-1].content
@tool
def weather_tool(city: str, dates: str) -> str:
    """Météo pour planifier voyage."""
    response = weather_agent.invoke({"messages": [HumanMessage(f"Météo {city} {dates} ? Impact voyage.")]})
    return response["messages"][-1].content

@tool
def booking_tool(trip_details: str) -> str:
    """Vérifiez disponibilités vols/hôtels."""
    response = booking_agent.invoke({"messages": [HumanMessage(f"Disponibilités : {trip_details}")]})
    return response["messages"][-1].content

@tool
def budget_tool(expenses: str, budget: str) -> str:
    """Optimisez budget voyage."""
    response = budget_agent.invoke({"messages": [HumanMessage(f"Budget {budget}, dépenses: {expenses} → optimisez.")]})
    return response["messages"][-1].content


orchestrator=create_agent(
    model=model,
    system_prompt="""Vous êtes un coordinateur de voyages. 
    Lors de la planification de voyages, utilisez les deux spécialistes : 
    1. Utilisez l'agent_plan_logistique pour calculer les détails pratiques : distances, temps, coûts et itinéraires 
    . 2. Utilisez l'agent_recommandations pour suggérer des attractions, des restaurants et des activités. 
    Combinez toujours les aspects pratiques et les recommandations intéressantes dans votre réponse finale.""",
    tools=[plan_logistics_agent, get_recommendations_agent, weather_tool, booking_tool, budget_tool]
)
# Test 1 : Planification d'un voyage en ville - utilise les deux agents
#response = orchestrator.invoke({
#    "messages": [HumanMessage(
#        "Planifiez un voyage de 3 jours à Rome. Je viens de Londres avec un budget de 2000€. "
#        "Calculez les coûts et le temps de trajet, et suggérez des attractions et restaurants incontournables."
#    )]
#})
#print(response["messages"][-1].content)

# Test 2 : Itinéraire multi-villes - les deux agents collaborent
#response = orchestrator.invoke({
#    "messages": [HumanMessage(
#        "Je souhaite visiter Paris, Amsterdam et Berlin en 7 jours au départ de New York. "
#        "Planifiez la logistique (vols, trains, coûts) et recommandez-moi "
#        "les 3 activités incontournables dans chaque ville."
#    )]
#})
#print(response["messages"][-1].content)

# Test 3 : Escapade de week-end - réponse de planification complète
#response = orchestrator.invoke({
#    "messages": [HumanMessage(
#        "Planifiez un week-end à Barcelone depuis Madrid. Budget : 500€. "
#        "Calculez le temps de trajet et les coûts, et suggérez les meilleurs endroits à visiter, "
#        "où manger et découvrir la culture locale."
#    )]
#})
#print(response["messages"][-1].content)

#print("=== TEST 1 : Voyage + Météo + Budget ===")
#response = orchestrator.invoke({
#    "messages": [HumanMessage(
#        "Planifiez 5 jours Paris depuis Antananarivo, budget 2500€. "
#        "Vérifiez météo, hôtels disponibles, optimisez budget."
#    )]
#})
#print(response["messages"][-1].content)

#print("\n=== TEST 2 : Multi-villes + Réservations ===")
#response = orchestrator.invoke({
#    "messages": [HumanMessage(
#        "Paris → Amsterdam → Berlin 7 jours depuis New York."
#        "Vols/trains disponibles ? Top 3 activités/ville ?"
#    )]
#})
#print(response["messages"][-1].content)

print("\n=== TEST 3 : Week-end + Validation ===")
response = orchestrator.invoke({
    "messages": [HumanMessage(
        "Week-end Barcelone depuis Madrid, 500€. Temps trajet, visites, culture locale, vérifiez faisabilité."
    )]
})
print(response["messages"][-1].content)
