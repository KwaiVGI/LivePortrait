# coding: utf-8

"""
face detectoin and alignment using InsightFace
"""

import numpy as np
from .rprint import rlog as log
from .dependencies.insightface.app import FaceAnalysis
from .dependencies.insightface.app.common import Face
from .timer import Timer


def sort_by_direction(faces, direction: str = 'large-small', face_center=None):
    if len(faces) <= 0:
        return faces

    if direction == 'left-right':
        return sorted(faces, key=lambda face: face['bbox'][0])
    if direction == 'right-left':
        return sorted(faces, key=lambda face: face['bbox'][0], reverse=True)
    if direction == 'top-bottom':
        return sorted(faces, key=lambda face: face['bbox'][1])
    if direction == 'bottom-top':
        return sorted(faces, key=lambda face: face['bbox'][1], reverse=True)
    if direction == 'small-large':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]))
    if direction == 'large-small':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]), reverse=True)
    if direction == 'distance-from-retarget-face':
        return sorted(faces, key=lambda face: (((face['bbox'][2]+face['bbox'][0])/2-face_center[0])**2+((face['bbox'][3]+face['bbox'][1])/2-face_center[1])**2)**0.5)
    return faces


class FaceAnalysisDIY(FaceAnalysis):
    def __init__(self, name='buffalo_l', root='~/.insightface', allowed_modules=None, **kwargs):
        super().__init__(name=name, root=root, allowed_modules=allowed_modules, **kwargs)

        self.timer = Timer()

    def get(self, img_bgr, **kwargs):
        max_num = kwargs.get('max_face_num', 0)  # the number of the detected faces, 0 means no limit
        flag_do_landmark_2d_106 = kwargs.get('flag_do_landmark_2d_106', True)  # whether to do 106-point detection
        direction = kwargs.get('direction', 'large-small')  # sorting direction
        face_center = None

        bboxes, kpss = self.det_model.detect(img_bgr, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname == 'detection':
                    continue

                if (not flag_do_landmark_2d_106) and taskname == 'landmark_2d_106':
                    continue

                # print(f'taskname: {taskname}')
                model.get(img_bgr, face)
            ret.append(face)

        ret = sort_by_direction(ret, direction, face_center)
        return ret

    def warmup(self):
        self.timer.tic()

        img_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
        self.get(img_bgr)

        elapse = self.timer.toc()
        log(f'FaceAnalysisDIY warmup time: {elapse:.3f}s')
