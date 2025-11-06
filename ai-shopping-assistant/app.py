import os
import requests
import json
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Using the standard, stable model which supports search grounding
GEMINI_MODEL_NAME = "gemini-2.5-flash" 

# --- Gemini API Setup ---
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set or loaded.")

# --- Flask Setup ---
app = Flask(__name__)

# --- Helper Functions ---

def call_gemini_api(payload):
    """Reusable function to call the Gemini API for text."""
    try:
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
        response = requests.post(
            apiUrl, 
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload)
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract the raw text response
        text_string = result.get('candidates')[0]['content']['parts'][0]['text']
        return text_string # Return raw text response

    except requests.exceptions.HTTPError as e:
        print(f"API HTTP Error: {e}")
        error_message = f"Gemini API HTTP Error: {e.response.status_code}. Details: {e.response.text}"
        return {"error": error_message}
    except Exception as e:
        print(f"An unexpected error occurred during AI generation: {e}")
        return {"error": f"An unexpected error occurred during AI generation: {e}"}

def generate_recommendations(prompt):
    """Generates the shopping recommendations as Markdown text with Google Search grounding."""
    system_instruction = (
        "You are an expert shopping assistant. Use the Google Search tool to find the most current products, pricing, and details. "
        "Your final response MUST be a single, well-formatted Markdown list of 3 product recommendations. "
        "For each recommendation, use an H3 heading (###) for the Product Name, followed by a detailed Description, and the Current Estimated Cost."
    )
    
    payload = {
        "contents": [{ "parts": [{ "text": prompt }] }],
        "systemInstruction": { "parts": [{ "text": system_instruction }] },
        "tools": [{ "google_search": {} }] # Google Search Grounding Enabled
    }
    return call_gemini_api(payload)

def generate_comparison(recommendations_markdown):
    """Generates a side-by-side comparison table."""
    comparison_prompt = (
        "Analyze the following list of product recommendations. Select the top two distinct products from this list. "
        "Generate a detailed, two-column Markdown table that compares the two products based on key features, price, and pros/cons. "
        "The comparison must be easy to read and structured as a Markdown table. "
        "Product List:\n\n" + recommendations_markdown
    )
    
    system_instruction = (
        "You are a product analyst. Your response MUST be a single, detailed Markdown table comparing the two best products from the input."
    )
    
    payload = {
        "contents": [{ "parts": [{ "text": comparison_prompt }] }],
        "systemInstruction": { "parts": [{ "text": system_instruction }] },
        "tools": [{ "google_search": {} }] 
    }
    return call_gemini_api(payload)

def generate_price_trend(prompt):
    """Generates a price trend analysis and purchase recommendation as a Markdown block."""
    system_instruction = (
        "You are a market analyst. Based on general market trends, seasonal sales cycles, and product launch patterns available via Google Search, "
        "provide a summary of the likely price trend for this product category over the next 60 days and give a clear purchase recommendation. "
        "Your response MUST be a concise Markdown block. Start with a clear heading (H3) for the recommendation, followed by a two-sentence summary of the trend."
    )
    
    payload = {
        "contents": [{ "parts": [{ "text": prompt }] }],
        "systemInstruction": { "parts": [{ "text": system_instruction }] },
        "tools": [{ "google_search": {} }] # Grounding required for market analysis
    }
    return call_gemini_api(payload)

def generate_product_personas(recommendations_markdown):
    """Generates a short persona/profile for each product in the list."""
    persona_prompt = (
        "Analyze the following list of product recommendations. For each product, create a short, catchy 'Product Persona' "
        "that summarizes the item's key appeal and target user in a memorable, two-sentence phrase. "
        "Your response MUST be a single Markdown list with one bolded persona description per product using the product name as the heading."
        "Product List:\n\n" + recommendations_markdown
    )
    
    system_instruction = (
        "You are a creative marketing strategist. Your response must be a single Markdown list of personas."
    )
    
    payload = {
        "contents": [{ "parts": [{ "text": persona_prompt }] }],
        "systemInstruction": { "parts": [{ "text": system_instruction }] },
        "tools": [{ "google_search": {} }]
    }
    return call_gemini_api(payload)

# FINAL FIX: Function to generate lowest price tracker (Markdown output)
def generate_price_tracker(query):
    """Finds the lowest price and retailer for the product category."""
    system_instruction = (
        "You are a deal finder. Use Google Search to find the current price and retailer for the product category specified by the user. "
        "List the three lowest prices currently available from major online retailers. "
        "Your response MUST be a clear Markdown list of the best 3 current deals, including the retailer name and the price details (e.g., $299 on Amazon)."
    )

def ask_product_ai(context, query):
    """Answers follow-up questions about products using the provided context."""
    system_instruction = (
        "You are a helpful shopping assistant. Answer the user's question based on the product context provided. "
        "Your response MUST be a clear, concise Markdown-formatted answer."
    )
    
    payload = {
        "contents": [{ "parts": [{ "text": f"Context: {context}\n\nQuestion: {query}" }] }],
        "systemInstruction": { "parts": [{ "text": system_instruction }] },
        "tools": [{ "google_search": {} }]
    }
    return call_gemini_api(payload)
    
    payload = {
        "contents": [{ "parts": [{ "text": f"Find the three lowest prices for: {query}" }] }],
        "systemInstruction": { "parts": [{ "text": system_instruction }] },
        "tools": [{ "google_search": {} }] # Critical: Must use search for live price data
    }
    # Return Markdown text now
    return call_gemini_api(payload)


# --- Flask Routes ---

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    data = request.json
    shopping_query = data.get('shopping_query')
    budget = data.get('budget')

    if not all([shopping_query, budget]):
        return jsonify({"error": "Missing form data"}), 400

    # 1. Construct the Prompts
    recommendation_prompt = (
        f"Find product recommendations for: '{shopping_query}'. "
        f"The user's budget level is '{budget}'. "
        "Find the best 3 options that match this request."
    )
    
    price_trend_prompt = (
        f"Analyze the market for products matching the description: '{shopping_query}'."
    )

    print(f"Sending grounded prompt to Gemini for Recommendations.")
    
    # 2. Call the AI Models 
    recommendations_markdown = generate_recommendations(recommendation_prompt)
    
    if isinstance(recommendations_markdown, dict) and "error" in recommendations_markdown:
        return jsonify(recommendations_markdown), 500

    comparison_markdown = generate_comparison(recommendations_markdown)
    if isinstance(comparison_markdown, dict) and "error" in comparison_markdown:
        comparison_markdown = "Comparison failed, but recommendations were successful."

    persona_markdown = generate_product_personas(recommendations_markdown)
    if isinstance(persona_markdown, dict) and "error" in persona_markdown:
        persona_markdown = "### Persona Analysis Unavailable"

    price_trend_markdown = generate_price_trend(price_trend_prompt)
    if isinstance(price_trend_markdown, dict) and "error" in price_trend_markdown:
        price_trend_markdown = "### Price Trend Analysis Unavailable\n\nCould not access live market data for price trend prediction. Please check retail sites for current sales."

    # FINAL FIX: Generate Price Tracker Data as Markdown
    price_tracker_markdown = generate_price_tracker(shopping_query)
    if isinstance(price_tracker_markdown, dict) and "error" in price_tracker_markdown:
        price_tracker_markdown = "Could not find a current price breakdown from major retailers."


    # 3. Return the data
    return jsonify({
        "recommendations_markdown": recommendations_markdown,
        "comparison_markdown": comparison_markdown,
        "price_trend_markdown": price_trend_markdown,
        "persona_markdown": persona_markdown,
        "price_tracker_markdown": price_tracker_markdown # NOW MARKDOWN
    })

# Route for Follow-up Questions (unchanged)
@app.route('/ask_product_ai', methods=['POST'])
def ask_ai_route():
    data = request.json
    context = data.get('context')
    query = data.get('query')

    if not context or not query:
        return jsonify({"error": "Missing context or query."}), 400

    answer_markdown = ask_product_ai(context, query)

    if isinstance(answer_markdown, dict) and "error" in answer_markdown:
        return jsonify(answer_markdown), 500
    
    return jsonify({"answer_markdown": answer_markdown})


if __name__ == '__main__':
    app.run(debug=True)