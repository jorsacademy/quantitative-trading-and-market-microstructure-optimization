from limit_order_book_simulator.engine import LimitOrderBook


def test_price_time_priority_at_same_price():
    book = LimitOrderBook()

    book.submit_limit(
        order_id="maker-1",
        trader_id="m1",
        side="sell",
        price_tick=101,
        quantity=5,
        timestamp=0,
    )
    book.submit_limit(
        order_id="maker-2",
        trader_id="m2",
        side="sell",
        price_tick=101,
        quantity=5,
        timestamp=1,
    )

    trades = book.submit_market(
        order_id="taker",
        trader_id="buyer",
        side="buy",
        quantity=7,
        timestamp=2,
    )

    assert [trade.maker_order_id for trade in trades] == [
        "maker-1",
        "maker-2",
    ]
    assert [trade.quantity for trade in trades] == [5, 2]
    assert book.order("maker-1") is None
    assert book.order("maker-2").remaining == 3


def test_better_price_executes_before_worse_price():
    book = LimitOrderBook()

    book.submit_limit(
        order_id="ask-102",
        trader_id="a",
        side="sell",
        price_tick=102,
        quantity=4,
        timestamp=0,
    )
    book.submit_limit(
        order_id="ask-101",
        trader_id="b",
        side="sell",
        price_tick=101,
        quantity=4,
        timestamp=1,
    )

    trades = book.submit_market(
        order_id="buy",
        trader_id="taker",
        side="buy",
        quantity=6,
        timestamp=2,
    )

    assert [trade.price_tick for trade in trades] == [101, 102]
    assert [trade.quantity for trade in trades] == [4, 2]


def test_marketable_limit_partially_fills_and_rests():
    book = LimitOrderBook()

    book.submit_limit(
        order_id="ask",
        trader_id="maker",
        side="sell",
        price_tick=101,
        quantity=3,
        timestamp=0,
    )

    trades = book.submit_limit(
        order_id="buy-limit",
        trader_id="buyer",
        side="buy",
        price_tick=102,
        quantity=7,
        timestamp=1,
    )

    assert sum(trade.quantity for trade in trades) == 3
    resting = book.order("buy-limit")
    assert resting is not None
    assert resting.remaining == 4
    assert resting.price_tick == 102
    assert book.best_bid_tick == 102


def test_cancel_removes_remaining_quantity():
    book = LimitOrderBook()
    book.submit_limit(
        order_id="bid",
        trader_id="maker",
        side="buy",
        price_tick=99,
        quantity=9,
        timestamp=0,
    )

    assert book.cancel("bid") == 9
    assert book.order("bid") is None
    assert book.best_bid_tick is None


def test_queue_ahead_reports_visible_priority():
    book = LimitOrderBook()
    for i, qty in enumerate((4, 6, 8), start=1):
        book.submit_limit(
            order_id=f"bid-{i}",
            trader_id=f"m{i}",
            side="buy",
            price_tick=99,
            quantity=qty,
            timestamp=i,
        )

    assert book.queue_ahead("bid-1") == 0
    assert book.queue_ahead("bid-2") == 4
    assert book.queue_ahead("bid-3") == 10
