import csv
import random
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class CrisisEngine:
    def __init__(self):
        # 1. Load Model
        # We use the CPU-optimized model path
        cache_path = os.path.join(os.getcwd(), "model_cache")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_path)
        
        # 2. Load Data using CSV (Saves ~80MB RAM vs Pandas)
        self.data = []
        csv_path = 'data/train.csv'
        
        try:
            with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
                # Reads file line by line without loading the whole thing into RAM
                reader = csv.DictReader(f)
                
                # Convert to list to sample
                all_rows = [row for row in reader if row.get('text')]
                
                # Randomly sample 200 items to keep memory usage low
                if len(all_rows) > 200:
                    self.data = random.sample(all_rows, 200)
                else:
                    self.data = all_rows
                    
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.data = []

        # 3. Process Data (The AI Part)
        if self.data:
            # Extract text list for the AI model
            texts = [row['text'] for row in self.data]
            
            # Convert text to numbers (Vectors)
            self.vectors = self.model.encode(texts, show_progress_bar=False)
            
            # Clustering (Find 5 themes)
            self.kmeans = KMeans(n_clusters=5, random_state=42)
            self.clusters = self.kmeans.fit_predict(self.vectors)
            
            # Dimensionality Reduction (384 dims -> 2 dims for the map)
            self.pca = PCA(n_components=2)
            self.coords = self.pca.fit_transform(self.vectors)
        else:
            self.vectors = []
            self.clusters = []
            self.coords = []
        
    def get_dashboard_data(self):
        """
        Returns data formatted exactly like the frontend expects.
        """
        results = []
        if not self.data:
            return results

        for i, row in enumerate(self.data):
            # Handle missing locations gracefully
            loc = row.get('location')
            if not loc or loc == "":
                loc = "Unknown"

            results.append({
                "id": i, 
                "text": row['text'],
                "location": loc,
                "cluster_id": int(self.clusters[i]),
                "pca_x": float(self.coords[i][0]), 
                "pca_y": float(self.coords[i][1]),
            })
        
        return results