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
            c.data[0, :] = np.arange(10)
            assert c.data[0, :].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            c.chunk_id[0] = 123456
            assert c.chunk_id[0] == 123456

    def test_no_accidental_reassigning(self):
        with forcing.ChunkCache.create(slots=3, chunksize=10) as c:
            with pytest.raises(AttributeError):
                c.chunk_id = [0, 1, 2]
