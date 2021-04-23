"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from irlc.assignments.assignment3_rl import week8bandits, Assignment3RL, week9dp, week10mc, week11nstep, week12linear, week13dqn
from unitgrade.unitgrade_helpers import evaluate_report_student

if __name__ == "__main__":
    # Run exercises for a single week: (Change as desired)
    week_to_run = 7

    # You can safely ignore the following
    w = {7: week8bandits,
         9: week9dp,
         10: week10mc,
         11: week11nstep,
         12: week12linear,
         13: week13dqn}
    Assignment3RL.title += f" (week {week_to_run})"
    Assignment3RL.questions = w[week_to_run]
    evaluate_report_student(Assignment3RL())
