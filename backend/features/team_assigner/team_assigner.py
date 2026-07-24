from sklearn.cluster import KMeans
import math
from collections import Counter

class TeamAssigner:
    def __init__(self):
        self.team_color = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def get_jersey_color(self, frame, bbox):
        image = frame[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])]
        jersey = image[0: int(image.shape[0]/2), :]
        kmeans = self.get_clustering_model(jersey)
        labels = kmeans.labels_
        clustered_image = labels.reshape(jersey.shape[0], jersey.shape[1])
        corner_cluster = [clustered_image[0,0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)
        player_cluster = 1 - non_player_cluster
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color
    
    def assign_team_color(self, frame, player_detections):
        player_color = []
        for _, player_detection in player_detections.items():
            if player_detection.get("is_goalkeeper"):
                continue
            bbox = player_detection['bbox']
            jersey_color = self.get_jersey_color(frame, bbox)
            player_color.append(jersey_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(player_color)
        self.kmeans = kmeans
        self.team_color[1] = kmeans.cluster_centers_[0]
        self.team_color[2] = kmeans.cluster_centers_[1]
    
    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_jersey_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id+=1
        self.player_team_dict[player_id] = team_id
        return team_id

    

    def assign_goalkeeper_team(self, goalkeeper_bbox, player_tracks, k=5):
        """
        Assign the goalkeeper's team based on the majority team
        among the k nearest outfield players.
        """

        gx = (goalkeeper_bbox[0] + goalkeeper_bbox[2]) / 2
        gy = (goalkeeper_bbox[1] + goalkeeper_bbox[3]) / 2

        neighbours = []

        for player in player_tracks.values():

            if player.get("is_goalkeeper"):
                continue

        # Ignore players without a team
            if "team" not in player:
                continue

            x1, y1, x2, y2 = player["bbox"]

            px = (x1 + x2) / 2
            py = (y1 + y2) / 2

            distance = math.hypot(px - gx, py - gy)

            neighbours.append((distance, player["team"]))

        if not neighbours:
            return None

        neighbours.sort(key=lambda x: x[0])

        # nearest = neighbours[:k]
        nearest = sorted(neighbours, key=lambda x: x[0])[:k]

        votes = Counter(team for _, team in nearest)

        return votes.most_common(1)[0][0]