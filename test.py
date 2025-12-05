print("--- Starting Test Script ---")

try:
    from app.ai_engine import CrisisEngine
    print("Import successful. Initializing Engine...")
    
    engine = CrisisEngine()
    print("Engine Initialized.")
    
    data = engine.get_dashboard_data()
    print(f"Data Retrieved. First Item: {data[0]}")

except Exception as e:
    print(f"❌ CRASHED: {e}")