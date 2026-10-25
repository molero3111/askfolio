from app.rate_limit import BookingDecision, BookingLimiter, Decision, MessageLimiter

DAY = 86400.0
T0 = 1_700_000_000.0 - (1_700_000_000.0 % DAY) + 3600  # 01:00 UTC


class FakeClock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now


def make_msg(clock, minute=5, day=50, glob=500):
    return MessageLimiter(minute, day, glob, clock=clock)


def test_minute_limit_then_notify_once_then_silent():
    lim = make_msg(FakeClock())
    assert [lim.check("a").decision for _ in range(5)] == [Decision.ALLOWED] * 5
    assert lim.check("a").decision == Decision.LIMITED_NOTIFY
    assert lim.check("a").decision == Decision.LIMITED_SILENT


def test_minute_window_slides():
    clock = FakeClock()
    lim = make_msg(clock)
    for _ in range(6):
        lim.check("a")
    clock.now += 61
    assert lim.check("a").decision == Decision.ALLOWED


def test_day_limit_resets_at_utc_midnight():
    clock = FakeClock()
    lim = make_msg(clock, minute=100, day=3)
    assert [lim.check("a").decision for _ in range(3)] == [Decision.ALLOWED] * 3
    assert lim.check("a").decision == Decision.LIMITED_NOTIFY
    assert lim.check("a").decision == Decision.LIMITED_SILENT
    clock.now = T0 - 3600 + DAY + 1  # just past next UTC midnight
    assert lim.check("a").decision == Decision.ALLOWED


def test_global_day_limit_across_chats():
    lim = make_msg(FakeClock(), glob=3)
    assert lim.check("a").decision == Decision.ALLOWED
    assert lim.check("b").decision == Decision.ALLOWED
    assert lim.check("c").decision == Decision.ALLOWED
    assert lim.check("d").decision == Decision.LIMITED_NOTIFY
    assert lim.check("d").decision == Decision.LIMITED_SILENT


def test_global_cap_notifies_each_chat_once():
    lim = make_msg(FakeClock(), glob=1)
    assert lim.check("a").decision == Decision.ALLOWED
    assert lim.check("b").decision == Decision.LIMITED_NOTIFY
    assert lim.check("b").decision == Decision.LIMITED_SILENT
    assert lim.check("c").decision == Decision.LIMITED_NOTIFY


def test_minute_window_boundary_exactly_60s():
    clock = FakeClock()
    lim = make_msg(clock)
    for _ in range(5):
        lim.check("a")
    clock.now += 59.9
    assert lim.check("a").decision == Decision.LIMITED_NOTIFY
    clock.now = T0 + 60
    assert lim.check("a").decision == Decision.ALLOWED


def test_day_roll_drops_idle_chat_state():
    clock = FakeClock()
    lim = make_msg(clock)
    for _ in range(6):
        lim.check("idle")
    clock.now = T0 - 3600 + DAY + 1
    lim.check("other")
    assert "idle" not in lim._recent
    assert ("minute", "idle") not in lim._notified


def test_booking_chat_cap():
    lim = BookingLimiter(1, 5, clock=FakeClock())
    lim.record("a")
    assert lim.can_book("a") == BookingDecision.CHAT_CAP
    assert lim.can_book("b") == BookingDecision.ALLOWED


def test_booking_global_cap():
    lim = BookingLimiter(1, 5, clock=FakeClock())
    for i in range(5):
        lim.record(f"c{i}")
    assert lim.can_book("c6") == BookingDecision.GLOBAL_CAP


def test_can_book_does_not_count():
    lim = BookingLimiter(1, 5, clock=FakeClock())
    for _ in range(3):
        lim.can_book("a")
    assert lim.record("a") == (1, 1)


def test_booking_resets_next_utc_day():
    clock = FakeClock()
    lim = BookingLimiter(1, 5, clock=clock)
    lim.record("a")
    clock.now = T0 - 3600 + DAY + 1
    assert lim.can_book("a") == BookingDecision.ALLOWED
    assert lim.record("a") == (1, 1)


def test_scope_reported_for_each_refusal():
    lim = make_msg(FakeClock(), minute=1, day=2, glob=3)
    assert lim.check("a").scope is None
    assert lim.check("a").scope == "minute"
    lim2 = make_msg(FakeClock(), minute=100, day=1, glob=100)
    lim2.check("a")
    assert lim2.check("a").scope == "chat_day"
    lim3 = make_msg(FakeClock(), glob=1)
    lim3.check("a")
    assert lim3.check("b").scope == "global"


def test_refused_calls_do_not_create_state():
    lim = make_msg(FakeClock(), glob=0)
    assert lim.check("ghost").decision == Decision.LIMITED_NOTIFY
    assert "ghost" not in lim._recent
    assert "ghost" not in lim._chat_day
