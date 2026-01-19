import numpy as np
import typing
from pyproj import CRS, Transformer
import xarray as xr
from scipy.ndimage import map_coordinates
from abc import abstractmethod


class Grid:
    DEFAULT_URL = "https://thredds.met.no/thredds/dodsC/fou-hi/norkystv3_800m_m00_be"

    def __init__(self, config) -> None:
        self.url = config['gridforce'].get('input_file', self.DEFAULT_URL)
        self._cache = {}
        self.xmin = -180
        self.xmax = 180
        self.ymin = -90
        self.ymax = 90

    def sample_metric(self, x, y):
        dy = np.broadcast_to(111_000.0, np.shape(x))
        dx = dy * np.cos(y*np.pi / 180)
        return dx, dy

    def sample_depth(self, x, y):
        assert np.shape(x) == np.shape(y)
        return np.broadcast_to(10_000.0, np.shape(x))

    def ll2xy(self, lon, lat):
        return lon, lat

    def ingrid(self, x, y):
        assert np.shape(x) == np.shape(y)
        return np.broadcast_to(True, np.shape(x))

    def atsea(self, x, y):
        i, j = self.xy2ij(x, y)
        return interp_nearest(self.mask, i, j)

    @property
    def mask(self) -> np.ndarray:
        if 'mask' not in self._cache:
            dset = DatasetWrapper.from_url(self.url)
            h = dset.get_data(field='h', idx=...)
            minimal_depth = np.min(h)
            self._cache['mask'] = h > minimal_depth
        return self._cache['mask']

    def _set_crs_cache(self):
        dset = DatasetWrapper.from_url(self.url)

        self._cache['crs'] = get_crs(dset)
        self._cache['transformer'] = Transformer.from_crs(
            crs_from=CRS.from_epsg(4326),
            crs_to=self._cache['crs'],
        )
        xi = dset.get_data(field='X', idx=...)
        eta = dset.get_data(field='Y', idx=...)
        xi_diff = np.diff(xi)
        eta_diff = np.diff(eta)
        resolution = xi_diff[0]
        assert np.all(xi_diff == resolution)
        assert np.all(eta_diff == resolution)
        self._cache['resolution'] = float(resolution)
        self._cache['xi_0'] = xi[0]
        self._cache['eta_0'] = eta[0]
            
    @property
    def crs(self) -> CRS:
        if 'crs' not in self._cache:
            self._set_crs_cache()
        c = self._cache['crs']
        assert isinstance(c, CRS)
        return c

    @property
    def transformer(self) -> Transformer:
        if 'transformer' not in self._cache:
            self._set_crs_cache()
        c = self._cache['transformer']
        assert isinstance(c, Transformer)
        return c

    @property
    def resolution(self) -> float:
        if 'resolution' not in self._cache:
            self._set_crs_cache()
        c = self._cache['resolution']
        assert isinstance(c, float)
        return c

    def xy2ij(self, x, y):
        if 'xi_0' not in self._cache or 'eta_0' not in self._cache:
            self._set_crs_cache()
        xi0 = self._cache['xi_0']
        eta0 = self._cache['eta_0']
        xi, eta = self.transformer.transform(y, x)
        return (eta - eta0) / self.resolution, (xi - xi0) / self.resolution


class Forcing:
    def __init__(self, config, grid):
        self.time_start = np.datetime64(config['start_time'], 's').astype('int64')
        self.time_step = np.int64(config['dt'])
        self.time_current = np.iinfo(np.int64).min

        self._grid = grid
        self.url = config['gridforce']['input_file']
        self.depth = config['gridforce']['depth']

        self.fields = {}
        self.fields['u_eastward'] = [np.zeros((0, 0), dtype='float32'), ] * 2
        self.fields['v_northward'] = [np.zeros((0, 0), dtype='float32'), ] * 2
        self.fields_tidx = np.iinfo(np.int64).min

        self._cache = {}

    def update(self, t):
        self.time_current = self.time_start + self.time_step * t
        
        i = np.searchsorted(self.forcing_times, self.time_current, side='right')
        tidx_current = np.clip(i - 1, 0, len(self.forcing_times) - 2)

        if tidx_current == self.fields_tidx:
            return
        
        elif tidx_current == self.fields_tidx + 1:
            idx1 = (tidx_current + 1, self.depth_index, ...)

            dset = DatasetWrapper.from_url(self.url)
            for f in self.fields:
                self.fields[f][0] = self.fields[f][1]
                data = dset.get_data(field=f, idx=idx1)
                self.fields[f][1] = np.nan_to_num(data)

            self.fields_tidx = tidx_current
        
        else:
            idx0 = (tidx_current, self.depth_index, ...)
            idx1 = (tidx_current + 1, self.depth_index, ...)

            dset = DatasetWrapper.from_url(self.url)
            for f in self.fields:
                data_0 = dset.get_data(field=f, idx=idx0)
                data_1 = dset.get_data(field=f, idx=idx1)
                self.fields[f][0] = np.nan_to_num(data_0)
                self.fields[f][1] = np.nan_to_num(data_1)

            self.fields_tidx = tidx_current

    def _load_small_fields(self):
        dset = DatasetWrapper.from_url(self.url)
        t_data = dset.get_data(field='time', idx=...)
        depth_data = dset.get_data(field='depth', idx=...)

        t = t_data.astype('datetime64[s]').astype('int64')
        self._cache['forcing_times'] = t
        self._cache['forcing_depths'] = depth_data
        self._cache['depth_idx'] = self._cache['forcing_depths'].tolist().index(self.depth)

    @property
    def depth_index(self):
        if 'depth_idx' not in self._cache:
            self._load_small_fields()
        return self._cache['depth_idx']

    @property
    def forcing_times(self):
        if 'forcing_times' not in self._cache:
            self._load_small_fields()
        return self._cache['forcing_times']

    def velocity(self, x, y, z, tstep):
        time = self.time_start + self.time_step * tstep
        
        t0 = self.forcing_times[self.fields_tidx]
        t1 = self.forcing_times[self.fields_tidx + 1]
        td = np.clip((time - t0) / (t1 - t0), 0, 1)

        i, j = self._grid.xy2ij(x, y)
        u0 = map_coordinates(self.fields['u_eastward'][0], (i, j), order=1, prefilter=False)
        u1 = map_coordinates(self.fields['u_eastward'][1], (i, j), order=1, prefilter=False)
        v0 = map_coordinates(self.fields['v_northward'][0], (i, j), order=1, prefilter=False)
        v1 = map_coordinates(self.fields['v_northward'][1], (i, j), order=1, prefilter=False)

        u = u0 * (1 - td) + u1 * td
        v = v0 * (1 - td) + v1 * td

        return u, v

    def close(self):
        pass



def load_field(url, field, idx: typing.Any = ...) -> np.ndarray:
    dset = DatasetWrapper.from_url(url)
    return dset.get_data(field=field, idx=idx)


def get_crs(dset: "DatasetWrapper") -> CRS:
    for v in dset.varnames():
        attrs = dset.get_attrs(v)
        if 'grid_mapping_name' in attrs:
            return CRS.from_cf(attrs)
    
    raise ValueError('CRS variable not found')


def interp_nearest(field, i, j):
    shp = np.shape(field)
    i = np.clip(np.round(i).astype(int), 0, shp[0] - 1)
    j = np.clip(np.round(j).astype(int), 0, shp[1] - 1)
    return field[i, j]


class DatasetWrapper:
    @staticmethod
    def from_url(url):
        url = str(url)
        if (url.startswith('http') or url.startswith('dap')) and ('://' in url):
            return DatasetWrapperOpendap(url)
        else:
            return DatasetWrapperXarrayFile(url)

    @abstractmethod
    def get_data(self, field, idx) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def get_attrs(self, field) -> dict:
        raise NotImplementedError
    
    @abstractmethod
    def varnames(self) -> list[str]:
        raise NotImplementedError


class DatasetWrapperXarrayFile(DatasetWrapper):
    def __init__(self, fname):
        super().__init__()
        self._fname = fname

    def get_data(self, field, idx) -> np.ndarray:
        with xr.open_dataset(self._fname) as dset:
            return dset[field][idx].to_numpy()
    
    def get_attrs(self, field) -> dict:
        with xr.open_dataset(self._fname) as dset:
            return dset[field].attrs
    
    def varnames(self) -> list[str]:
        with xr.open_dataset(self._fname) as dset:
            return [str(v) for v in dset.variables]


class DatasetWrapperOpendap(DatasetWrapper):
    def __init__(self, fname):
        super().__init__()
        self._fname = fname

    def get_data(self, field, idx) -> np.ndarray:
        with xr.open_dataset(self._fname, engine='pydap') as dset:
            return dset[field][idx].to_numpy()
    
    def get_attrs(self, field) -> dict:
        with xr.open_dataset(self._fname, engine='pydap') as dset:
            return dset[field].attrs
    
    def varnames(self) -> list[str]:
        with xr.open_dataset(self._fname, engine='pydap') as dset:
            return [str(v) for v in dset.variables]
