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
        artifact_suffix=['.py', '.pth', '.json']),
]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
