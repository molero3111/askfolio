from dataclasses import dataclass, field
from enum import Enum

_MAX_HISTORY = 10


class SchedulingMode(str, Enum):
    IDLE = "idle"
    AWAITING_SLOT_SELECTION = "awaiting_slot_selection"
    AWAITING_RECRUITER_INFO = "awaiting_recruiter_info"
    CANCELLED = "cancelled"


@dataclass
class ConversationState:
    mode: SchedulingMode = SchedulingMode.IDLE
    history: list[dict] = field(default_factory=list)
    proposed_slots: list[dict] = field(default_factory=list)
    selected_slot: dict | None = None
    recruiter_name: str | None = None
    recruiter_email: str | None = None
    recruiter_display_tz: str | None = None  # IANA key if recruiter requested a specific tz

    def add_message(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > _MAX_HISTORY:
            self.history = self.history[-_MAX_HISTORY:]

    def reset_scheduling(self) -> None:
        self.mode = SchedulingMode.IDLE
        self.proposed_slots = []
        self.selected_slot = None
        self.recruiter_name = None
        self.recruiter_email = None
        self.recruiter_display_tz = None

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "history": self.history,
            "proposed_slots": self.proposed_slots,
            "selected_slot": self.selected_slot,
            "recruiter_name": self.recruiter_name,
            "recruiter_email": self.recruiter_email,
            "recruiter_display_tz": self.recruiter_display_tz,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConversationState":
        obj = cls()
        obj.mode = SchedulingMode(d.get("mode", SchedulingMode.IDLE.value))
        obj.history = d.get("history", [])
        obj.proposed_slots = d.get("proposed_slots", [])
        obj.selected_slot = d.get("selected_slot")
        obj.recruiter_name = d.get("recruiter_name")
        obj.recruiter_email = d.get("recruiter_email")
        obj.recruiter_display_tz = d.get("recruiter_display_tz")
        return obj
