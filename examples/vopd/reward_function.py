def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Return a neutral reward for OPD-only smoke runs.

    The OPD smoke experiments use teacher-student distillation as the training
    signal. verl still invokes a reward function during rollout, so this keeps
    unsupported datasets such as MathVista from failing before OPD loss is used.
    """
    return 0.0
