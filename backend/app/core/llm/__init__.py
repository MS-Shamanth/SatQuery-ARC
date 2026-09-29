"""The language model layer, behind one small interface.

Two jobs in this system use a language model, and neither of them touches a
measurement. Drafting an Analysis Contract reads the question and proposes a plan,
which the validator then checks against the registry and repairs. Narration takes a
finished evidence ledger and phrases it, after which every figure in the wording is
audited against the ledger and the whole sentence is discarded if one does not
match. That is the entire remit, and it is why swapping providers is a
configuration change rather than a rewrite: nothing downstream trusts the model.

The interface is deliberately narrow. A provider is handed messages and a schema
and returns text, or it raises. It does not know what a contract is. Adding a
provider means implementing :class:`LlmProvider` and naming it in ``LLM_PROVIDER``.

Nothing here logs a key or an Authorization header. The redaction is not left to
the caller's discretion: :func:`redact` is applied to provider error text before it
is raised, because the most likely way a key escapes is inside an exception
message that someone then logs.
"""

from app.core.llm.base import (
    LlmError,
    LlmProvider,
    LlmRateLimited,
    LlmResponse,
    LlmUnavailable,
    Message,
    redact,
)
from app.core.llm.registry import (
    available_providers,
    get_provider,
    reset_provider_for_tests,
)

__all__ = [
    "LlmError",
    "LlmProvider",
    "LlmRateLimited",
    "LlmResponse",
    "LlmUnavailable",
    "Message",
    "available_providers",
    "get_provider",
    "redact",
    "reset_provider_for_tests",
]
