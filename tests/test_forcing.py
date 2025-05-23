from ladim.gridforce import ROMS
import numpy as np
from ladim import forcing
import pytest


class Test_z2s:
    def test_returns_interpolated_s_level(self):
        zrho = np.array([-5, -4, -3, -2]).reshape((4, 1, 1))
        k, a = ROMS.z2s(zrho, np.zeros(5), np.zeros(5), np.array([6, 5, 3.5, 2, 0]))
        assert k.tolist() == [1, 1, 2, 3, 3]
        assert a.tolist() == [1.0, 1.0, 0.5, 0.0, 0.0]


class Test_timestring_formatter:
    def test_can_format_simple_date(self):
        result = forcing.timestring_formatter(
            pattern="My time: {time:%Y-%m-%d %H:%M:%S}",
            time="2012-12-31T23:58:59",
        )
        assert result == "My time: 2012-12-31 23:58:59"

    def test_can_format_shifted_dates(self):
        result = forcing.timestring_formatter(
            pattern="My time: {time - 3600:%Y-%m-%d %H:%M:%S}",
            time="2012-12-31T23:58:59",
        )
        assert result == "My time: 2012-12-31 22:58:59"


class Test_ChunkCache:
    def test_creates_correct_header(self):
        with forcing.ChunkCache.create(slots=3, chunksize=10) as c:
            assert c.mem.size > 0
            assert c.num_chunks == 3
            assert c.chunksize == 10
            assert c.datatype == b'float32'
            assert c.itemsize == 4
            assert len(c.chunk_id) == 3
            assert c.data.shape == (3, 10)
            assert str(c.data.dtype) == 'float32'

    def test_can_push_pull_data(self):
        with forcing.ChunkCache.create(slots=3, chunksize=10) as c:
            c.push(data=np.arange(10), id=12345)
            c.push(data=np.arange(10, 20), id=12346)
            assert c.pull(12346).tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            assert c.pull(12345).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_can_evict_older_chunks(self):
        with forcing.ChunkCache.create(slots=3, chunksize=10) as c:
            # Push single chunk, the data is there
            c.push(data=np.arange(10), id=12345)
            assert c.contains(12345)

            # Push three more chunks, the first one is evicted in the last step
            c.push(data=np.arange(10)+1, id=12346)
            assert c.contains(12345)
            c.push(data=np.arange(10)+2, id=12347)
            assert c.contains(12345)
            c.push(data=np.arange(10)+3, id=12348)
            assert not c.contains(12345)

    def test_no_accidental_reassigning(self):
        with forcing.ChunkCache.create(slots=3, chunksize=10) as c:
            with pytest.raises(AttributeError):
                c.chunk_id = [0, 1, 2]


class Test_update_lru:
    def test_update_lru_moves_slot_to_front(self):
        # Test moving middle element
        lru = np.array([0, 1, 2, 3, 4], dtype=np.int16)
        forcing.update_lru(lru, 2)
        assert lru.tolist() == [2, 0, 1, 3, 4]

        # Test moving the first element (should be no change)
        lru = np.array([0, 1, 2, 3, 4], dtype=np.int16)
        forcing.update_lru(lru, 0)
        assert lru.tolist() == [0, 1, 2, 3, 4]

        # Test moving the last element
        lru = np.array([0, 1, 2, 3, 4], dtype=np.int16)
        forcing.update_lru(lru, 4)
        assert lru.tolist() == [4, 0, 1, 2, 3]

        # Test moving an element that is not in the array (wrong behavior, but no error or infinite loop)
        lru = np.array([0, 1, 2, 3, 4], dtype=np.int16)
        forcing.update_lru(lru, 5)
        assert lru.tolist() == [5, 0, 1, 2, 3]