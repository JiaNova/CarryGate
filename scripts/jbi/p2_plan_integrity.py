"""Integrity rules for the frozen P2 plan and its registered appendices."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


FROZEN_P2_PLAN_BODY_SHA256 = "0272f75ad815411f98d49d37f0792b1780e72eef6ed922b04f5db2dec9b72c72"
P2_C1_RESOLUTION_HEADING = "\n## 附：C1 训练目标的澄清".encode("utf-8")
P2_C1_RESOLUTION_APPENDIX_SHA256 = "4364fb6203fde156375763972db62f8622681e3cc8e37dbd6840a24cdce03c63"
P2_MEMORY_NOTE_HEADING = "\n## 附：显存实现说明".encode("utf-8")
P2_MEMORY_NOTE_APPENDIX_SHA256 = "d873e10374e2cb7eb0263bc8558e8eb7083f04d07f0196862d5107a03a0a3f26"
P2_TEST_INFERENCE_OOM_HEADING = "\n## 执行偏离（2026-10-11）：P2 测试推理 OOM 后按既有 padded-size 规则分块".encode("utf-8")
P2_TEST_INFERENCE_OOM_APPENDIX_SHA256 = "f30848088ba648571ac253501df8aea0ae15afb24e96c83ebb6dceee88a2cf5c"
# Appendices dated after the freeze, in the order they were appended. Each
# runs from its heading to the next registered heading (or end of file) and
# is hashed on its own. Execution-only deviations do not alter the frozen
# analysis-plan document identity.
REGISTERED_APPENDICES = (
    ("c1_resolution", P2_C1_RESOLUTION_HEADING, P2_C1_RESOLUTION_APPENDIX_SHA256),
    ("memory_implementation_note", P2_MEMORY_NOTE_HEADING, P2_MEMORY_NOTE_APPENDIX_SHA256),
    ("test_inference_oom_deviation", P2_TEST_INFERENCE_OOM_HEADING, P2_TEST_INFERENCE_OOM_APPENDIX_SHA256),
)


def inspect_plan_identity(path: Path) -> dict[str, Any]:
    """Verify the original design body and each exact registered appendix.

    The analysis-plan hash recorded before training identifies the frozen
    design body. Appendices must appear in registration order, each at most
    once, with nothing unregistered after the body.
    """
    raw = path.read_bytes()
    found = []
    for name, heading, expected in REGISTERED_APPENDICES:
        count = raw.count(heading)
        if count > 1:
            raise RuntimeError(f"P2 plan contains duplicate {name} appendices")
        if count == 1:
            found.append((raw.index(heading), name, expected))
    if [name for _, name, _ in sorted(found)] != [name for name, _, _ in REGISTERED_APPENDICES[:len(found)]]:
        raise RuntimeError("P2 plan appendices are missing or out of registration order")
    found.sort()
    body = raw[:found[0][0]] if found else raw
    appendix_shas: dict[str, str] = {}
    for position, (start, name, expected) in enumerate(found):
        end = found[position + 1][0] if position + 1 < len(found) else len(raw)
        digest = hashlib.sha256(raw[start + 1:end]).hexdigest()
        if digest != expected:
            raise RuntimeError(f"P2 {name} appendix differs from its registered text")
        appendix_shas[name] = digest
    body_sha = hashlib.sha256(body).hexdigest()
    if body_sha != FROZEN_P2_PLAN_BODY_SHA256:
        raise RuntimeError("frozen P2 plan body changed after sample and overlap audit")
    deviation_start = next(
        (start for start, name, _ in found if name == "test_inference_oom_deviation"),
        len(raw),
    )
    return {
        "analysis_plan_sha256": body_sha,
        "analysis_plan_document_sha256": hashlib.sha256(raw[:deviation_start]).hexdigest(),
        "c1_resolution_appendix_sha256": appendix_shas.get("c1_resolution"),
        "execution_deviation_appendix_sha256": appendix_shas.get("test_inference_oom_deviation"),
        "registered_appendix_sha256": appendix_shas,
    }
