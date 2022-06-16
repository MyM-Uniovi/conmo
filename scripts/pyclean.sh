#!/bin/bash
#
# Script Name: pyclean
#
# Version: 0.1   
#
# Author: perezlucas@uniovi.es
# Date : 25/05/2022
#
# Description: Deletes the cache files from Commo.

find ../. | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
