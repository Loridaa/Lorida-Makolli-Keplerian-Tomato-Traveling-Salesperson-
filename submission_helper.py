""" Submission helper to generate a .json-file for submission to optimize.esa.int.
(Same format as the official helper: https://api.optimize.esa.int/data/tools/submission_helper.py)
"""
import json
import numpy


def create_submission(challenge_id, problem_id, x, fn_out='./submission.json', name='', description=''):
    x = numpy.array(x, dtype=object).tolist()
    d = {'decisionVector': x,
         'problem': problem_id,
         'challenge': challenge_id,
         'name': name,
         'description': description}
    with open(fn_out, 'wt') as json_file:
        json.dump([d], json_file, indent=6)
