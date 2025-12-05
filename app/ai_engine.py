import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

class CrisisEngine:
    def __init__(self):
        
        #load the model
        cache_path = os.path.join(os.getcwd(), "model_cache")
        if os.path.exists(cache_path):
             self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_path)
        else:
             self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        #load the crisis data
        # Ensure data/train.csv exists
        try:
            self.df = pd.read_csv('data/train.csv')
            # For simulation, let's take a sample so it's fast
            self.df = self.df.sample(n=min(200, len(self.df)), random_state=42).reset_index(drop=True)
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.df = pd.DataFrame(columns=['text', 'location']) # Empty fallback

        
        if not self.df.empty:
            #pre_calculate vectors
            self.vectors = self.model.encode(self.df['text'].tolist(), show_progress_bar=True)
            
            #Perform Clustering
            ## We tell it to find 5 "Themes" in the data
            self.kmeans = KMeans(n_clusters=5, random_state=42)
            self.clusters = self.kmeans.fit_predict(self.vectors)
            
            #Perform PCA (Dimensionality Reduction for Visualization)
            #Squash 384 dims -> 2 dims (x, y)
            self.pca = PCA(n_components=2)
            self.coords = self.pca.fit_transform(self.vectors)
        else:
            self.vectors = []
            self.clusters = []
            self.coords = []
        
    def get_dashboard_data(self):
        """
        Returns data ready for the Frontend Map & Charts
        """
        results = []
        if self.df.empty:
            return results

        for i, row in self.df.iterrows():
            results.append({
                "id": int(i),
                "text": row['text'],
                "location": row['location'] if pd.notna(row['location']) else "Unknown",
                "cluster_id": int(self.clusters[i]),
                "pca_x": float(self.coords[i][0]), 
                "pca_y": float(self.coords[i][1]),
            })
        
        return results