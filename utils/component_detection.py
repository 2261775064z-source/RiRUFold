"""Explicit centroid Pd / unmatched-component Fa, with component IDs tracked."""
import cv2
import numpy as np


def components(mask):
    _, _, stats, centers = cv2.connectedComponentsWithStatsWithAlgorithm(
        np.asarray(mask, np.uint8), 8, cv2.CV_32S, cv2.CCL_WU)
    return stats[1:, cv2.CC_STAT_AREA], centers[1:]


class ComponentDetection:
    def __init__(self):
        self.hits = self.targets = self.false_pixels = self.pixels = 0

    def update(self, prediction, gt):
        previous_hits = self.hits
        areas, centers = components(prediction)
        _, targets = components(gt)
        used = np.zeros(len(areas), bool)
        for center in targets:
            matches = np.flatnonzero((np.linalg.norm(centers - center, axis=1) < 3) & ~used)
            if len(matches):
                used[matches[0]] = True
                self.hits += 1
        self.targets += len(targets)
        false_pixels = int(areas[~used].sum())
        self.false_pixels += false_pixels
        self.pixels += gt.size
        return dict(detected_targets=self.hits - previous_hits, total_targets=len(targets),
                    false_component_pixels=false_pixels)

    def result(self):
        return dict(Pd=self.hits / max(self.targets, 1), Fa=self.false_pixels / max(self.pixels, 1),
                    detected_targets=self.hits, total_targets=self.targets,
                    false_component_pixels=self.false_pixels, total_pixels=self.pixels,
                    definition='8-connected components; first unused centroid within distance<3; unmatched predicted area / all pixels')
