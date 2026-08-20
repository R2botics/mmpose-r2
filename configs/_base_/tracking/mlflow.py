# MLflow tracking backend. Include this in any config that should log to MLflow:
#
#   _base_ = ['./your-config.py', '../_base_/tracking/mlflow.py']
#
# Or set MLFLOW_TRACKING_URI in the environment to redirect to a server.

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SafeMLflowVisBackend',
        exp_name='box-flap-pose',
        tracking_uri='http://10.200.104.2:31016',
        # MUST be a tuple: mmengine's scandir() rejects a list with
        # '"suffix" must be a string or tuple of strings'. MLflowVisBackend
        # calls it inside close(), so a list makes close() raise — which
        # aborts artifact upload (this is why .pth files never reached MLflow)
        # and skips mlflow.end_run(), leaving runs stuck in RUNNING.
        artifact_suffix=('.py', '.pth', '.json')),
]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
