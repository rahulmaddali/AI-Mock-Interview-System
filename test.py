import google.generativeai as genai

# Replace with your new API key
API_KEY = "AIzaSyAR2vvXBVbTTGPPu1HuITm3_WLSUczxeMA"

# Configure the API
genai.configure(api_key=API_KEY)

# Preferred model
primary_model = "models/gemini-1.5-pro-latest"
fallback_model = "models/gemini-1.5-flash-latest"

try:
    # Test primary model
    print(f"✅ Using model: {primary_model}")
    model = genai.GenerativeModel(primary_model)
    response = model.generate_content("Hello from Gemini Pro!")
    print("Response:", response.text)

except Exception as e:
    if "quota" in str(e).lower() or "429" in str(e):
        print("⚠️ Pro quota exceeded. Switching to Flash model...")
        try:
            model = genai.GenerativeModel(fallback_model)
            print(f"✅ Using fallback model: {fallback_model}")
            response = model.generate_content("Hello from Gemini Flash!")
            print("Response:", response.text)
        except Exception as fe:
            print("❌ Fallback model also failed:", fe)
    else:
        print("❌ API call failed:", e)