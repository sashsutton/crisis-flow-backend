from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np


class CrisisEngine:
    def __init__(self):
        
        #load the model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        #load the crisis data
        self.df = pd.read_csv('data/train.csv')
        
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
    def get_dashboard_data(self):
        """
        Returns data ready for the Frontend Map & Charts
        """
        results = []
        for i, row in self.df.iterrows():
            results.append({
                "id": int(i),
                "text": row['text'],
                "location": row['location'] if pd.notna(row['location']) else "Unknown",
                "cluster_id": int(self.clusters[i]),
                "pca_x": self.coords[i][0],
                "pca_y": self.coords[i][1],
            })
            return
        
        
        