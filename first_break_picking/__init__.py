# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc.
#
# This file is part of the First-Break Picking package.
# Licensed under the terms of the MIT License.
#
# https://github.com/geo-stack/first_break_picking
# =============================================================================

import os
import sys
package_path = os.path.abspath(os.path.join(__file__, "../"))
sys.path.append(package_path)

# from first_break_picking.train_eval import unet as unet
from first_break_picking.train_eval.train import train
from first_break_picking.train_eval.predict import predict
from first_break_picking.train_eval.predict import Predictor
