import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

class CrisisEngine:
    def __init__(self):
        # 1. Load the Model
        # We check for a local cache first
        cache_path = os.path.join(os.getcwd(), "model_cache")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_path)
        
        # 2. Load Data with Pandas
        # We read the CSV and randomly sample 200 rows for the dashboard
        try:
            self.df = pd.read_csv('data/train.csv')
            # Ensure we don't crash if the file is smaller than 200 rows
            sample_n = min(200, len(self.df))
            self.df = self.df.sample(n=sample_n, random_state=42).reset_index(drop=True)
            
            # Fill missing locations
            self.df['location'] = self.df['location'].fillna('Unknown')
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.df = pd.DataFrame(columns=['text', 'location'])

        # 3. Process Data (AI Analysis)
        if not self.df.empty:
            # Convert text to vectors (The heavy lifting)
            self.vectors = self.model.encode(self.df['text'].tolist(), show_progress_bar=False)
            
            # Clustering: Group messages by similarity
            self.kmeans = KMeans(n_clusters=5, random_state=42)
            self.clusters = self.kmeans.fit_predict(self.vectors)
            
            # Dimensionality Reduction: Squash 384 dimensions to 2 for the plot
            self.pca = PCA(n_components=2)
            self.coords = self.pca.fit_transform(self.vectors)
        else:
            self.vectors = []
            self.clusters = []
            self.coords = []
        
    def get_dashboard_data(self):
        """
        Format data for the React Frontend
        """
        results = []
        if self.df.empty:
            return results

        for i, row in self.df.iterrows():
            results.append({
                "id": int(i),
                "text": row['text'],
                "location": row['location'],
                "cluster_id": int(self.clusters[i]),
                "pca_x": float(self.coords[i][0]), 
                "pca_y": float(self.coords[i][1]),
            })
        
        return results