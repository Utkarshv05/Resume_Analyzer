import pickle

MODEL_PATH = r"D:\finalYear\Resume-Screening-App-main\clf.pkl"

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

print("✅ Model loaded successfully!")
