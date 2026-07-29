import numpy as np
import torch
import supervision as sv
from PIL import Image
from more_itertools import chunked
from sklearn.cluster import KMeans
from transformers import AutoProcessor, SiglipVisionModel
import umap

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


class TeamClassifier:
    """
    Embeddings-based team classifier using Siglip + UMAP + KMeans.
    Mirrors the concept shown in the notebook.
    """

    def __init__(self, device='cuda', n_clusters=2, batch_size=32):
        self.device = device
        self.n_clusters = n_clusters
        self.batch_size = batch_size

        self.embeddings_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
        self.embeddings_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)

        self.reducer = umap.UMAP(n_components=3)
        self.clustering_model = KMeans(n_clusters=n_clusters)
        self.fitted = False

    # ------------------------------------------------------------------ #
    def _ensure_pil(self, images):
        """Convert OpenCV/numpy crops to PIL if necessary."""
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(sv.cv2_to_pillow(img))
            elif isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
        return pil_images

    # ------------------------------------------------------------------ #
    def _extract_embeddings(self, crops):
        crops = self._ensure_pil(crops)
        batches = chunked(crops, self.batch_size)
        data = []

        with torch.no_grad():
            for batch in batches:
                inputs = self.embeddings_processor(images=batch, return_tensors='pt').to(self.device)
                outputs = self.embeddings_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().detach().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    # ------------------------------------------------------------------ #
    def fit(self, crops):
        data = self._extract_embeddings(crops)
        projections = self.reducer.fit_transform(data)
        self.clustering_model.fit(projections)
        self.fitted = True

    # ------------------------------------------------------------------ #
    def predict(self, crops):
        if not self.fitted:
            raise RuntimeError("TeamClassifier must be fitted before predict().")
        data = self._extract_embeddings(crops)
        projections = self.reducer.transform(data)
        return self.clustering_model.predict(projections)


# ---------------------------------------------------------------------- #
def resolve_goalkeepers_teamid(player_detections, gk_detections):
    """
    Assign goalkeeper team IDs based on proximity to team centroids.
    """
    if len(player_detections) == 0 or len(gk_detections) == 0:
        return np.array([])

    gk_xy = gk_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    player_xy = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    team_0_mask = player_detections.class_id == 0
    team_1_mask = player_detections.class_id == 1

    # Fallback if one team isn't present
    if not np.any(team_0_mask) or not np.any(team_1_mask):
        return np.zeros(len(gk_detections), dtype=int)

    team_0_centroid = player_xy[team_0_mask].mean(axis=0)
    team_1_centroid = player_xy[team_1_mask].mean(axis=0)

    gk_team_ids = []
    for gkxy in gk_xy:
        dist_0 = np.linalg.norm(gkxy - team_0_centroid)
        dist_1 = np.linalg.norm(gkxy - team_1_centroid)
        gk_team_ids.append(0 if dist_0 < dist_1 else 1)

    return np.array(gk_team_ids)