import pytest

from levanter.data._process_interleave import InProgressSequence


@pytest.mark.asyncio
async def test_append():
    seq = InProgressSequence[int]()
    seq.append(1)
    assert seq.current_length() == 1
    assert await seq.get(0) == 1


@pytest.mark.asyncio
async def test_set_item():
    seq = InProgressSequence[int]()
    seq.set_item(2, 10)
    assert seq.current_length() == 3
    assert await seq.get(2) == 10


@pytest.mark.asyncio
async def test_set_item_out_of_range():
    seq = InProgressSequence[int]()
    with pytest.raises(IndexError):
        seq.set_item(-1, 10)


@pytest.mark.asyncio
async def test_item_exception():
    seq = InProgressSequence[int]()
    seq.set_item(0, 5)
    seq.item_exception(0, ValueError("Test Exception"))
    with pytest.raises(ValueError, match="Test Exception"):
        await seq.get(0)


@pytest.mark.asyncio
async def test_set_finished_length():
    seq = InProgressSequence[int]()
    seq.append(1)
    seq.append(2)
    seq.set_finished_length(2)
    assert seq.is_finished()
    assert seq.to_list() == [1, 2]


@pytest.mark.asyncio
async def test_set_finished_length_first():
    seq = InProgressSequence[int]()
    seq.set_finished_length(2)
    seq.append(1)
    seq.append(2)
    assert seq.is_finished()
    assert seq.to_list() == [1, 2]


@pytest.mark.asyncio
async def test_finalize():
    seq = InProgressSequence[int]()
    seq.append(1)
    seq.append(2)
    seq.finalize()
    assert seq.is_finished()
    assert seq.to_list() == [1, 2]


@pytest.mark.asyncio
async def test_exception_handling():
    seq = InProgressSequence[int]()
    seq.set_exception(ValueError("Test Exception"))
    with pytest.raises(ValueError, match="Test Exception"):
        await seq.finished_promise


@pytest.mark.asyncio
async def test_get_promise_immediate():
    seq = InProgressSequence[int]()
    seq.append(1)
    promise = seq.get_promise(0)
    assert await promise == 1


@pytest.mark.asyncio
async def test_get_promise_deferred():
    seq = InProgressSequence[int]()
    promise = seq.get_promise(0)
    seq.append(2)
    assert await promise == 2


@pytest.mark.asyncio
async def test_get_promise_out_of_range():
    seq = InProgressSequence[int]()
    seq.set_finished_length(2)
    with pytest.raises(IndexError):
        seq.get_promise(3)


@pytest.mark.asyncio
async def test_get_promise_with_future_exception():
    seq = InProgressSequence[int]()
    promise = seq.get_promise(0)
    promise2 = seq.get_promise(0)
    seq.item_exception(0, ValueError("Test Exception"))

    with pytest.raises(ValueError, match="Test Exception"):
        await promise

    with pytest.raises(ValueError, match="Test Exception"):
        await promise2


@pytest.mark.asyncio
async def test_get_promise_with_past_exception():
    seq = InProgressSequence[int]()
    seq.item_exception(0, ValueError("Test Exception"))
    promise = seq.get_promise(0)
    promise2 = seq.get_promise(0)
    with pytest.raises(ValueError, match="Test Exception"):
        await promise

    with pytest.raises(ValueError, match="Test Exception"):
        await promise2
