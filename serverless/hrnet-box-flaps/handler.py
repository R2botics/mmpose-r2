import base64
import json
import cv2
import numpy as np

from mmpose.apis import init_model, inference_topdown

MODEL = None
KPT_NAMES = ['NORTH_1', 'NORTH_2', 'EAST_1', 'EAST_2',
             'SOUTH_1', 'SOUTH_2', 'WEST_1', 'WEST_2']
KPT_THR = 0.3


def init_context(context):
    global MODEL
    config     = '/opt/nuclio/hrnet-w32_8-kp.py'
    checkpoint = '/opt/nuclio/best_coco_AP_epoch_30.pth'
    device     = 'cpu'
    MODEL = init_model(config, checkpoint, device=device)
    context.logger.info('HRNet model loaded')


def handler(context, event):
    data = event.body
    if isinstance(data, (bytes, bytearray)):
        data = json.loads(data)

    img_bytes = base64.b64decode(data['image'])
    buf = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    results = inference_topdown(MODEL, img, bboxes=None)

    annotations = []
    for result in results:
        keypoints = result.pred_instances.keypoints[0]       # K x 2
        scores    = result.pred_instances.keypoint_scores[0] # K

        elements = [
            {
                'label':   KPT_NAMES[i],
                'points':  [round(float(kp[0]), 2), round(float(kp[1]), 2)],
                'type':    'points',
                'outside': bool(float(scores[i]) < KPT_THR),
            }
            for i, kp in enumerate(keypoints)
        ]

        annotations.append({
            'confidence': str(round(float(scores.mean()), 4)),
            'label':      'Box_Flaps',
            'type':       'skeleton',
            'elements':   elements,
        })

    return context.Response(
        body=json.dumps(annotations),
        headers={},
        content_type='application/json',
        status_code=200,
    )
