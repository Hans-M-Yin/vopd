import math
import re
import unicodedata


_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

_CN_UNITS = {"十": 10, "百": 100, "千": 1000}


def _extract_last_boxed(text):
    """Extract the content of the last \\boxed{...}, supporting nested braces."""
    if text is None:
        return ""
    text = str(text)
    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return text

    start = starts[-1]
    brace_start = text.find("{", start)
    depth = 0
    content_start = brace_start + 1
    for idx in range(brace_start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:idx]
    return text[content_start:]


def _parse_simple_chinese_number(text):
    text = text.strip()
    if not text or any(ch not in _CN_DIGITS and ch not in _CN_UNITS for ch in text):
        return None
    if len(text) == 1 and text in _CN_DIGITS:
        return _CN_DIGITS[text]

    total = 0
    current = 0
    used_unit = False
    for ch in text:
        if ch in _CN_DIGITS:
            current = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            total += (current or 1) * unit
            current = 0
            used_unit = True
    total += current
    return total if used_unit else None


def _normalize_text(text):
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKC", text)
    text = _extract_last_boxed(text)
    text = text.strip().lower()
    text = text.replace("\\,", "").replace("\\%", "%")
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"[\s，,。.;；:：!！?？'\"`~]", "", text)
    return text


def _extract_number(text):
    normalized = unicodedata.normalize("NFKC", "" if text is None else str(text))
    boxed = _extract_last_boxed(normalized).strip().replace("\\%", "%")
    boxed = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", boxed)

    cn_value = _parse_simple_chinese_number(boxed)
    if cn_value is not None:
        return float(cn_value)

    # Prefer the last number because model answers often end with the final result.
    matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:/\d+(?:\.\d*)?)?%?", boxed)
    if not matches:
        return None

    raw = matches[-1]
    is_percent = raw.endswith("%")
    raw = raw[:-1] if is_percent else raw
    try:
        if "/" in raw:
            numerator, denominator = raw.split("/", 1)
            value = float(numerator) / float(denominator)
        else:
            value = float(raw)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return value / 100.0 if is_percent else value


def _numeric_equal(pred, gt):
    pred_num = _extract_number(pred)
    gt_num = _extract_number(gt)
    if pred_num is None or gt_num is None:
        return False
    return math.isclose(pred_num, gt_num, rel_tol=1e-4, abs_tol=1e-4)


def _option_equal(pred, gt):
    pred_norm = _normalize_text(pred)
    gt_norm = _normalize_text(gt)
    option_pattern = re.compile(r"^[\(（\[]?([a-z])[\)）\].、]?$")
    pred_match = option_pattern.match(pred_norm)
    gt_match = option_pattern.match(gt_norm)
    return bool(pred_match and gt_match and pred_match.group(1) == gt_match.group(1))


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Rule-based reward for boxed final answers.

    The model is expected to put its final answer in \\boxed{...}. We compare the
    final boxed answer against the ground truth with numeric, option, and
    normalized-text matching.
    """
    pred = _extract_last_boxed(solution_str)
    gt = _extract_last_boxed(ground_truth)

    if _numeric_equal(pred, gt):
        return 1.0
    if _option_equal(pred, gt):
        return 1.0
    if _normalize_text(pred) == _normalize_text(gt):
        return 1.0
    return 0.0
