"""Per-segment state tracking for the annotation GUI."""

from enum import Enum

from speech_annotator_common.models import Segment


class SegmentState(Enum):
    """State machine for each annotation segment."""
    LLM_MERGED = "llm_merged"
    USER_EDITING = "user_editing"
    VALIDATED = "validated"


class SegmentData:
    """Tracks all data for a single segment during the annotation process."""

    def __init__(
        self,
        index: int,
        segment: Segment,
        merged_text: str,
        confidence: int,
        auto_texts: list[str],
    ) -> None:
        self.index = index
        self.segment = segment
        self.merged_text = merged_text
        self.confidence = confidence
        self.auto_texts = auto_texts
        self.current_text = merged_text
        self.state = SegmentState.LLM_MERGED
        self.edited = False
        self.word_edit_count = 0
        self.propagated = False
        self.baseline_text = merged_text

    def update_text(self, new_text: str) -> None:
        """Called when user edits the transcription text.

        A segment that was already validated stays validated: the
        annotator's edit is itself a verification act. Demoting it would
        silently drop the segment from the verified output file on the
        next save (with no way to re-validate it from the GUI).
        """
        self.current_text = new_text
        self.edited = True
        if self.state == SegmentState.VALIDATED:
            self.word_edit_count = _word_edit_distance(self.baseline_text, new_text)
        else:
            self.state = SegmentState.USER_EDITING

    def validate(self) -> None:
        """Called when user validates the segment."""
        self.state = SegmentState.VALIDATED
        self.word_edit_count = _word_edit_distance(self.baseline_text, self.current_text)

    def apply_propagation(self, new_text: str) -> None:
        """Apply LLM-propagated corrections without affecting user correction stats."""
        self.current_text = new_text
        self.baseline_text = new_text
        self.propagated = True

    @property
    def is_validated(self) -> bool:
        return self.state == SegmentState.VALIDATED


def _word_edit_distance(text_a: str, text_b: str) -> int:
    """Compute word-level Levenshtein distance between two texts."""
    words_a = text_a.split()
    words_b = text_b.split()
    if not words_a and not words_b:
        return 0
    if not words_a:
        return len(words_b)
    if not words_b:
        return len(words_a)

    # DP table for Levenshtein
    prev = list(range(len(words_b) + 1))
    for i in range(1, len(words_a) + 1):
        curr = [i] + [0] * len(words_b)
        for j in range(1, len(words_b) + 1):
            cost = 0 if words_a[i - 1] == words_b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost, # substitution
            )
        prev = curr
    return prev[-1]
