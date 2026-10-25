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
    assert [lim.check("a") for _ in range(5)] == [Decision.ALLOWED] * 5
    assert lim.check("a") == Decision.LIMITED_NOTIFY
    assert lim.check("a") == Decision.LIMITED_SILENT


def test_minute_window_slides():
    clock = FakeClock()
    lim = make_msg(clock)
    for _ in range(6):
        lim.check("a")
    clock.now += 61
    assert lim.check("a") == Decision.ALLOWED


def test_day_limit_resets_at_utc_midnight():
    clock = FakeClock()
    lim = make_msg(clock, minute=100, day=3)
    assert [lim.check("a") for _ in range(3)] == [Decision.ALLOWED] * 3
    assert lim.check("a") == Decision.LIMITED_NOTIFY
    assert lim.check("a") == Decision.LIMITED_SILENT
    clock.now = T0 - 3600 + DAY + 1  # just past next UTC midnight
    assert lim.check("a") == Decision.ALLOWED


def test_global_day_limit_across_chats():
    lim = make_msg(FakeClock(), glob=3)
    assert lim.check("a") == Decision.ALLOWED
    assert lim.check("b") == Decision.ALLOWED
    assert lim.check("c") == Decision.ALLOWED
    assert lim.check("d") == Decision.LIMITED_NOTIFY
    assert lim.check("e") == Decision.LIMITED_SILENT


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
