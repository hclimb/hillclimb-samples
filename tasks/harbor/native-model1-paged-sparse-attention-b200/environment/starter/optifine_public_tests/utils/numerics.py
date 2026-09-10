import torch

from utils.native_compare import check_is_allclose, get_cos_diff


def compare(answer, expected, tolerances):
    output, lse = answer
    expected_output, expected_lse = expected
    output_limits, lse_limits = tolerances
    report = {}
    for name, actual, reference, limits in (
        ('output', output, expected_output, output_limits),
        ('lse', lse, expected_lse, lse_limits),
    ):
        if actual.shape != reference.shape or actual.dtype != reference.dtype:
            raise AssertionError(f'{name}: shape/dtype {actual.shape}/{actual.dtype}, '
                                 f'expected {reference.shape}/{reference.dtype}')
        passed = check_is_allclose(name, actual, reference, *limits)
        finite = torch.isfinite(reference) & torch.isfinite(actual)
        actual_finite, reference_finite = actual[finite].float(), reference[finite].float()
        error = (actual_finite - reference_finite).abs()
        report[name] = dict(max_absolute=error.max().item() if error.numel() else 0,
                            distance=get_cos_diff(actual_finite, reference_finite), passed=passed)
        if not passed:
            raise AssertionError(f'{name}: numerical mismatch: {report[name]}')
    return report
