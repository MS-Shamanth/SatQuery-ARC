"""Language-model narration, tightly fenced.

The model's job here is wording and nothing else. It is handed the finished
ledger and asked to read it back in two or three sentences. It is not asked what
the answer is, it is not given the imagery, and it cannot introduce a quantity,
because every number it writes is checked against the ledger afterwards and an
explanation with an untraceable figure is discarded rather than corrected.

Keeping the model on this side of the line is what lets the rest of the system
claim that its numbers are measured. The narration is the one place a language
model touches the output, and it is the one place the output is audited.
"""

from __future__ import annotations

import logging

from app.config import get_settings
from app.models.verdict import VERDICT_LABEL_TEXT, EvidenceLedger, Verdict

logger = logging.getLogger(__name__)

NARRATION_TIMEOUT_SECONDS = 15.0

SYSTEM_INSTRUCTION = """
You phrase findings for a remote-sensing analyst. You are given a completed
analysis: a claim, a verdict, the measurements behind it, and the alternative
explanations that were tested.

Write two or three sentences of plain prose that read the finding back.

Rules, in order of importance:

1. Use only the numbers given to you, copied exactly as written including their
   units and the number of decimal places shown. Do not re-round them, do not
   convert units, do not compute new figures, and do not add one of your own. An
   explanation containing a number that is not in the input is discarded.
   Quote confidence as a percentage of the value given.
2. Do not change the verdict, soften it, or hedge it. If it says refuted, say
   refuted.
3. If an alternative explanation is still standing, say so and name it. That is
   the most important thing in the finding, not a footnote.
4. Write for someone who will act on this. No preamble, no restating the
   question, no offers to help further.
5. British spelling. No bullet points, no headings, no markdown.
""".strip()


def _brief(verdict: Verdict, ledger: EvidenceLedger) -> str:
    """The ledger, written out for the model. Numbers exactly as measured."""
    lines = [
        f"Claim under test: {verdict.claim}",
        f"Verdict: {VERDICT_LABEL_TEXT[verdict.label]}",
        f"Confidence: {verdict.confidence:.2f}",
        "",
        "Measurements admitted as evidence:",
    ]
    for item in ledger.items:
        lines.append(
            f"- {item.label}: {item.display} ({item.direction.value} "
            f"the claim; {item.relevance})"
        )

    if ledger.consistency:
        lines.append("")
        lines.append("Independent cross-checks:")
        for check in ledger.consistency:
            lines.append(
                f"- {check.quantity}: {check.first_value:g} against "
                f"{check.second_value:g} {check.unit}, "
                f"{'agreeing' if check.agrees else 'disagreeing'}"
            )

    lines.append("")
    lines.append("Alternative explanations tested:")
    for test in ledger.confounders:
        lines.append(
            f"- {test.label}: {test.verdict.value.replace('_', ' ')} "
            f"({test.measured})"
        )

    if verdict.requirements:
        lines.append("")
        lines.append("Data that would settle what is open:")
        for requirement in verdict.requirements:
            lines.append(f"- {requirement}")

    lines.append("")
    lines.append(f"The reasoning to paraphrase: {verdict.reasoning}")
    return "\n".join(lines)


def narrate_verdict(
    verdict: Verdict, ledger: EvidenceLedger
) -> tuple[str | None, str | None]:
    """Phrase a verdict. Returns (text, model) or (None, None) when unavailable.

    Provider-agnostic, and unaffected by which provider is configured, because the
    fence around this call does not depend on the model: the brief contains
    pre-formatted display strings, the instruction forbids arithmetic, and the
    caller audits every figure in the reply against the ledger and discards the
    whole sentence if one does not trace. A different model can only produce
    different wording, never a different number.

    Returning ``(None, None)`` is a supported outcome. The verdict's deterministic
    reasoning is always present, so a run with no narration is a run with plainer
    prose and identical content.
    """
    from app.core.llm import LlmError, Message, get_provider

    provider = get_provider()
    if not provider.configured:
        return None, None

    try:
        response = provider.complete(
            [
                Message(role="system", content=SYSTEM_INSTRUCTION),
                Message(role="user", content=_brief(verdict, ledger)),
            ],
            temperature=0.2,
            # Generous: some free models on OpenRouter reason before answering and
            # spend tokens doing it, and a truncated explanation is worse than none.
            max_tokens=2048,
        )
    except LlmError as exc:
        # Already redacted. Info rather than warning: losing the phrasing costs
        # nothing that matters, and the reasoning is unaffected.
        logger.info("narration unavailable: %s", str(exc)[:200])
        return None, None

    text = response.text.strip()
    return (text, response.model) if text else (None, None)


__all__ = ["SYSTEM_INSTRUCTION", "narrate_verdict"]
