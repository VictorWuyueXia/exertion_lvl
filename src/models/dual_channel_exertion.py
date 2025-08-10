# -*- coding: utf-8 -*-
"""
Dual Channel Exertion Model
Re-export from vgg16_exertion.py for compatibility
"""

from .vgg16_exertion import (
    DualChannelExertionModel,
    coral_predict,
    coral_predict_proba,
    coral_loss,
    coral_targets,
    CoralHead,
    VGG16ishMFCC,
    W2VTemporal,
    AttnPool,
    count_parameters,
    validate_config,
    create_model
)

__all__ = [
    'DualChannelExertionModel',
    'coral_predict',
    'coral_predict_proba', 
    'coral_loss',
    'coral_targets',
    'CoralHead',
    'VGG16ishMFCC',
    'W2VTemporal',
    'AttnPool',
    'count_parameters',
    'validate_config',
    'create_model'
]
