"""Gate policy: the thresholds and rules that turn NLI scores into a decision.

The policy is frozen and produces a stable ``policy_id`` that is written into
every ledger entry, so an auditor can detect a permissive configuration (e.g. a
lowered contradiction threshold) without re-running the gate.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GatePolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # A claim is "contradicted" if any citation contradicts it at >= this score.
    # Checked FIRST: contradiction always wins over entailment (fail-closed).
    tau_contra: float = Field(default=0.5, ge=0.0, le=1.0)
    # A claim is "entailed" if (not contradicted and) any citation entails it at
    # >= this score.
    tau_entail: float = Field(default=0.5, ge=0.0, le=1.0)
    # NLI-neutral is treated as "baseless" (unsupported -> deny). This is fixed
    # to a Literal so the only honest fail-closed reading is preserved; it exists
    # as a field purely so it is recorded in the policy_id.
    treat_neutral_as: Literal["baseless"] = "baseless"
    # An answer with fewer than this many citations is denied outright.
    min_citations: int = Field(default=1, ge=0)
    # A citation may only entail or contradict a claim if at least this fraction
    # of the claim's content words appears in it. Filters distractor passages
    # (which NLI models confidently mislabel) before aggregation. 0.0 restores
    # the original "any citation may vote" behavior. Default 0.2 was chosen
    # empirically: it left a curated set's accuracy unchanged while cutting the
    # false-deny rate on noisy-retrieval cases roughly 9x. See README.
    min_relevance: float = Field(default=0.2, ge=0.0, le=1.0)

    def policy_id(self) -> str:
        """Stable short id derived from the policy fields."""
        payload = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()
        return f"policy-{digest}"


DEFAULT_POLICY = GatePolicy()
