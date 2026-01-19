from ladim import makrel
import pytest
import numpy as np


@pytest.fixture(scope='function')
def conf0() -> dict:
    return dict(
        date='2000-01-01 01:02:03',
        num=2,
        location=[5, 60],
        seed=0,
    )


class Test_make_release:
    def test_returns_dict_of_lists(self, conf0):
        r = makrel.make_release(conf0)
        assert isinstance(r, dict)
        assert all(isinstance(v, list) for v in r.values())

    def test_returns_correct_keys(self, conf0):
        r = makrel.make_release(conf0)
        assert list(r.keys()) == ['date', 'longitude', 'latitude', 'depth']

    def test_returns_correct_number(self, conf0):
        r = makrel.make_release(conf0)
        assert all(len(v) == conf0['num'] for v in r.values())

    def test_accepts_stream(self, conf0):
        import io
        s = '{"date": "2000-01-01 01:02:03", "num": 2, "location": [5, 60]}'
        r1 = makrel.make_release(io.StringIO(s))
        r2 = makrel.make_release(conf0)
        assert r1 == r2

    def test_throws_FileNotFound_if_invalid_filename(self):
        with pytest.raises(FileNotFoundError):
            makrel.make_release("this_is_no_file_name")

    def test_can_add_numeric_attribute(self, conf0):
        conf0['attrs'] = dict(
            first=0,
        )
        r = makrel.make_release(conf0)
        assert all(k in r.keys() for k in conf0['attrs'].keys())
        assert r['first'] == [0, 0]

    def test_can_add_ranged_attribute(self, conf0):
        conf0['attrs'] = dict(
            second=[1, 2],
        )
        r = makrel.make_release(conf0)
        assert all(k in r.keys() for k in conf0['attrs'].keys())
        assert r['second'] == [1, 2]

    def test_can_add_callable_attribute(self, conf0):
        conf0['attrs'] = dict(
            third=lambda num: np.arange(num) + 10,
        )
        r = makrel.make_release(conf0)
        assert all(k in r.keys() for k in conf0['attrs'].keys())
        assert r['third'] == [10, 11]

    def test_can_add_stringfunction_attribute(self, conf0):
        conf0['attrs'] = dict(
            fourth="numpy.arange",
        )
        r = makrel.make_release(conf0)
        assert r['fourth'] == [0, 1]

    def test_can_add_gaussian_attribute(self, conf0):
        conf0['new_attr'] = dict(
            distribution='gaussian',
            mean=100,
            std=10,
        )
        r = makrel.make_release(conf0)
        assert r['new_attr'] == [117.64052345967664, 104.00157208367223]

    def test_can_add_exponential_attribute(self, conf0):
        conf0['new_attr'] = dict(
            distribution='exponential',
            mean=100,
        )
        r = makrel.make_release(conf0)
        assert r['new_attr'] == [79.587450816311, 125.59307629658379]

    def test_can_add_piecewise_random_attribute(self, conf0):
        conf0['new_attr'] = dict(
            distribution='piecewise',
            knots=[10, 11, 12, 13],
            cdf=[0, .1, .9, 1],
        )
        r = makrel.make_release(conf0)
        assert r['new_attr'] == [11.561016879909156, 11.768986707965524]

    def test_can_add_multiple_attributes(self, conf0):
        conf0['attrs'] = dict(
            first=0,
            second=[1, 2],
            third=lambda num: np.arange(num) + 10,
            fourth="numpy.arange",
        )

        r = makrel.make_release(conf0)

        assert all(k in r.keys() for k in conf0['attrs'].keys())
        assert r['first'] == [0, 0]
        assert r['second'] == [1, 2]
        assert r['third'] == [10, 11]
        assert r['fourth'] == [0, 1]

    def test_can_change_column_sortorder(self, conf0):
        conf0['attrs'] = dict(first=0, second=1, third=2)
        conf0['columns'] = ['longitude', 'latitude', 'first', 'third', 'depth']

        r = makrel.make_release(conf0)

        assert list(r.keys()) == conf0['columns']
        assert r['first'] == [0, 0]
        assert r['third'] == [2, 2]

    def test_accepts_numpy_date(self, conf0):
        r0 = makrel.make_release(conf0)
        conf0['date'] = np.datetime64(conf0['date'])
        r1 = makrel.make_release(conf0)
        assert r0 == r1

    def test_can_return_date_range(self, conf0):
        conf0['num'] = 7
        conf0['date'] = ['2000-01-01 01:00', '2000-01-01 02:00']
        r = makrel.make_release(conf0)
        assert r['date'] == [
            '2000-01-01T01:00:00',
            '2000-01-01T01:10:00',
            '2000-01-01T01:20:00',
            '2000-01-01T01:30:00',
            '2000-01-01T01:40:00',
            '2000-01-01T01:50:00',
            '2000-01-01T02:00:00',
        ]

    def test_can_set_depth(self, conf0):
        r = makrel.make_release(conf0)
        assert r['depth'] == [0, 0]

        conf0['depth'] = 3
        r = makrel.make_release(conf0)
        assert r['depth'] == [3, 3]

        conf0['depth'] = [3, 6]
        conf0['num'] = 3
        r = makrel.make_release(conf0)
        assert r['depth'] == [4.646440511781974, 5.145568099117258, 4.8082901282149315]

    def test_can_print_to_file(self, conf0):
        import io
        handle = io.StringIO()
        makrel.make_release(conf0, handle)

        handle.seek(0)
        assert handle.read().replace('\r', '') == (
            '2000-01-01T01:02:03\t5\t60\t0.0\n'
            '2000-01-01T01:02:03\t5\t60\t0.0\n'
        )

    def test_can_use_polygon_as_sampling_area(self, conf0):
        conf0['num'] = 5
        conf0['location'] = [[1, 0, 1], [0, 0, 1]]

        r = makrel.make_release(conf0)

        assert r['latitude'] == [
            0.3541058869333439,
            0.4375872112626925,
            0.10822699921792023,
            0.03633723949897072,
            0.3834415188257777,
        ]
        assert r['longitude'] == [
            0.5623808488506793,
            0.966482131015597,
            0.5401824381239879,
            0.11074060120630969,
            0.45447757702366465,
        ]
        assert all(lon >= lat for lon, lat in zip(r['longitude'], r['latitude']))

    def test_can_use_multipolygon_as_sampling_area(self, conf0):
        lat = [[60, 60, 61, 61], [70, 70, 71]]
        lon = [[5, 6, 6, 5], [10, 11, 11]]
        conf0['location'] = [lon, lat]
        conf0['num'] = 5

        result = makrel.make_release(conf0)
        assert result['latitude'] == [
            60.79172503808267,
            70.4375872112627,
            60.568044561093934,
            60.92559663829266,
            60.92896394180211,
        ]
        assert result['longitude'] == [
            5.56238084885068,
            10.966482131015598,
            5.540182438123988,
            5.1107406012063095,
            5.454477577023665
        ]

    def test_can_use_geojson_as_sampling_area(self, conf0):
        geojson = """
{
    "type": "FeatureCollection",
    "name": "test",
    "crs": {
        "type": "name",
        "properties": {
            "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
        }
    },
    "features": [
        {
            "type": "Feature",
            "properties": { "myprop": 101 },
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [ [ [ 5, 60 ], [ 5, 61 ], [ 6, 61 ], [ 6, 60 ], [ 5, 60 ] ] ],
                    [ [ [ 10, 70 ], [ 11, 70 ], [ 11, 71], [ 10, 70 ] ] ]
                ]
            }
        },
        {
            "type": "Feature",
            "properties": { "myprop": 102 },
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [ [ [ 2, 54 ], [ 2, 55 ], [ 3, 55 ], [ 3, 54 ], [ 2, 54 ] ] ]
                ]
            }
        }
    ]
}
"""
        import io
        geojson_file = io.StringIO(geojson)

        conf0['location'] = geojson_file
        conf0['num'] = 10
        result = makrel.make_release(conf0)

        assert result['latitude'] == [
            70.20827496191734,
            54.200841435783275,
            54.53852063774707,
            70.07440336170734,
            70.07103605819789,
            54.639921021327524,
            70.02021839744033,
            54.94466891704958,
            54.52184832175007,
            60.41466193999052
        ]
        assert result['longitude'] == [
            10.22965661968457,
            2.328053483969628,
            2.029523923346864,
            10.293874185420883,
            10.18931048406682,
            2.2729496789709356,
            10.163571684849373,
            2.222711237402478,
            2.699994927300078,
            5.715325911762657
        ]
        assert result['myprop'] == [101, 102, 102, 101, 101, 102, 101, 102, 102, 101]

    def test_can_use_offset_polygon_as_sampling_area(self, conf0):
        conf0['location'] = dict(
            center=[0, 60],
            offset=[
                [-50000, 50000, 50000, -50000],
                [-50000, -50000, 50000, 50000],
            ]
        )
        conf0['num'] = 5

        result = makrel.make_release(conf0)

        assert result['latitude'] == [
            60.26228062814893,
            59.974021504109096,
            60.061176683164156,
            60.382640291567434,
            59.6143322738132,
        ]
        assert result['longitude'] == [
            0.11207533991782959,
            0.8380960561199113,
            0.0721929966399305,
            -0.6993553348470959,
            0.08178697665959145,
        ]

    def test_accepts_list_of_configs(self, conf0):
        r = makrel.make_release([conf0, conf0])
        assert len(r['date']) == 4

    def test_sorts_output_by_release_date_when_multiple_groups(self):
        conf = dict(
            seed=0,
            groups=[
                dict(
                    date=['2000-01-01', '2000-01-31'],
                    num=10,
                    location=[5, 60],
                    attrs=dict(group_id=1),
                ),
                dict(
                    date=['2000-01-01', '2000-01-31'],
                    num=10,
                    location=[5, 60],
                    attrs=dict(group_id=2),
                ),
            ],
        )
        r = makrel.make_release(conf)
        dates = np.array(r['date']).astype('datetime64')
        seconds = (dates - dates[0]) / np.timedelta64(1, 's')
        assert all(np.diff(seconds) >= 0)

    def test_adds_attributes_within_geojson_file(self, conf0):
        geojson = """
        {"type": "FeatureCollection", "name": "test",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
        {"type": "Feature", "properties": { "cage_id": 1 }, "geometry": {"type": "MultiPolygon",
        "coordinates": [[ [ [ 0, 0 ], [ 0, 1 ], [ 1, 1 ], [ 1, 0 ], [ 0, 0 ] ] ],
                        [ [ [ 10, 10 ], [ 11, 10 ], [ 11, 11], [ 10, 10 ] ] ]] }},
        {"type": "Feature", "properties": { "cage_id": 2}, "geometry": {"type": "MultiPolygon",
        "coordinates": [[ [ [ 2, 4 ], [ 2, 5 ], [ 3, 5 ], [ 3, 4 ], [ 2, 4 ] ] ]] }} ]}
        """

        import io
        geojson_file = io.StringIO(geojson)

        conf0['location'] = geojson_file
        conf0['num'] = 10
        result = makrel.make_release(conf0)

        assert 'cage_id' in result
        assert np.all(np.isin(result['cage_id'], [1, 2]))


class Test_triangulate:
    def test_if_triangle(self):
        coords = np.array([[0, 0], [0, 1], [1, 0]])
        triangles = makrel.triangulate(coords)
        assert triangles.tolist() == [[[0, 0], [0, 1], [1, 0]]]

    def test_if_clockwise_square(self):
        coords = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        triangles = makrel.triangulate(coords)
        assert triangles.tolist() == [
            [[0, 0], [0, 1], [1, 1]],
            [[0, 0], [1, 1], [1, 0]],
        ]

    def test_if_counterclockwise_square(self):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        triangles = makrel.triangulate(coords)
        assert triangles.tolist() == [
            [[0, 0], [1, 0], [1, 1]],
            [[0, 0], [1, 1], [0, 1]],
        ]


class Test_triangle_areas:
    def test_if_single_triangle(self):
        coords = np.array([[0, 0], [0, 1], [1, 0]])
        area = makrel.triangle_areas(coords)
        assert area.tolist() == 0.5

    def test_if_multiple_triangles(self):
        coords = np.array([
            [[0, 0], [0, 1], [1, 0]],
            [[0, 0], [0, 2], [1, 0]],
        ])
        area = makrel.triangle_areas(coords)
        assert area.tolist() == [0.5, 1]


class Test_get_polygon_sample_convex:
    def test_all_inside_when_triangle(self):
        coords = np.array([[5, 3], [5, 1], [6, 3]])
        x, y = makrel.get_polygon_sample_convex(coords, 100)

        assert np.all(x >= 5)
        assert np.all(y <= 3)
        assert np.all(2 * (y - 1) >= (x - 5))

    def test_all_inside_when_rectangle(self):
        coords = np.array([[1, 10], [2, 10], [2, 12], [1, 12]])
        x, y = makrel.get_polygon_sample_convex(coords, 100)

        assert np.all(x >= 1)
        assert np.all(x <= 2)
        assert np.all(y >= 10)
        assert np.all(y <= 12)

    def test_does_not_work_when_nonconvex_polygon(self):
        coords = np.array([[0, 0], [10, 0], [10, 10], [9, 1]])
        x, y = makrel.get_polygon_sample_convex(coords, 100)
        is_inside_forbidden_area = (x < 9) & (y > 1)
        assert np.count_nonzero(is_inside_forbidden_area) > 0


class Test_get_polygon_sample_nonconvex:
    def test_all_inside_when_triangle(self):
        coords = np.array([[5, 3], [5, 1], [6, 3]])
        x, y = makrel.get_polygon_sample_nonconvex(coords, 100)

        assert np.all(x >= 5)
        assert np.all(y <= 3)
        assert np.all(2 * (y - 1) >= (x - 5))

    def test_all_inside_when_rectangle(self):
        coords = np.array([[1, 10], [2, 10], [2, 12], [1, 12]])
        x, y = makrel.get_polygon_sample_nonconvex(coords, 100)

        assert np.all(x >= 1)
        assert np.all(x <= 2)
        assert np.all(y >= 10)
        assert np.all(y <= 12)

    def test_works_when_nonconvex_polygon(self):
        coords = np.array([[0, 0], [10, 0], [10, 10], [9, 1]])
        x, y = makrel.get_polygon_sample_nonconvex(coords, 100)
        is_inside_forbidden_area = (x < 9) & (y > 1)
        assert np.count_nonzero(is_inside_forbidden_area) == 0


class Test_is_convex:
    def test_returns_true_if_clockwise_square(self):
        coords = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        assert makrel.is_convex(coords)

    def test_returns_true_if_counterclockwise_square(self):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert makrel.is_convex(coords)

    def test_returns_false_if_nonconvex_quadrilateral(self):
        coords = np.array([[0, 0], [1, 0], [.1, .1], [0, 1]])
        assert not makrel.is_convex(coords)


class Test_metric_degree_conversion:
    def test_metric_to_degrees_correct_when_lat_60(self):
        dx = 1000
        dy = 2000
        lat = 60
        dlon, dlat = makrel.metric_diff_to_degrees(dx, dy, lat)
        assert (dlon, dlat) == (0.01796630568239042, 0.017981358739225493)

    def test_degrees_to_metric_correct_when_lat_60(self):
        dlon = 0.02
        dlat = 0.01
        lat = 60
        dx, dy = makrel.degree_diff_to_metric(dlon, dlat, lat)
        assert (dx, dy) == (1113.194907932736, 1112.2629991453834)

    def test_back_and_forth_is_identity_when_lat_60(self):
        dx = 1000
        dy = 2000
        lat = 60
        dlon, dlat = makrel.metric_diff_to_degrees(dx, dy, lat)
        dx2, dy2 = makrel.degree_diff_to_metric(dlon, dlat, lat)

        assert np.isclose(dx, dx2)
        assert np.isclose(dy, dy2)


class Test_load_config:
    def test_throws_filenotfound_if_nonexisting_file(self):
        with pytest.raises(FileNotFoundError) as e:
            makrel.load_config('nofile')

        msg = str(e.value)
        assert msg == "[Errno 2] No such file or directory: 'nofile'"

    def test_throws_valueerror_if_wrong_yaml_format(self):
        import io
        buf = io.StringIO("*unknown_tag")
        with pytest.raises(ValueError) as e:
            makrel.load_config(buf)

        msg = str(e.value)
        assert msg == (
            "Error parsing yaml file: found undefined alias 'unknown_tag'\n"
            '  in "<unicode string>", line 1, column 1:\n'
            '    *unknown_tag\n'
            '    ^'
        )

    def test_throws_typeerror_if_wrong_format(self):
        with pytest.raises(TypeError) as e:
            makrel.load_config(123)

        msg = str(e.value)
        assert msg == "Not a valid config format: <class 'int'>"

    def test_throws_valueerror_if_missing_parameters(self):
        config = dict(
            groups=[
                dict(num=1, date=['2000-01-01', '2000-01-02'], location=[5, 9]),
                dict(num=1, date=['2000-01-01', '2000-01-02']),
                dict(num=1),
            ],
        )
        with pytest.raises(ValueError) as e:
            makrel.load_config(config)

        msg = str(e.value)
        assert msg == (
            'Missing parameters: location in group 1\n'
            '  and date, location in group 2'
        )

    def test_throws_valueerror_if_wrong_date_format(self):
        config = dict(num=1, date=['nodate', 'nodate2'], location=[5, 9])
        with pytest.raises(ValueError) as e:
            makrel.load_config(config)

        msg = str(e.value)
        assert msg == 'Error parsing datetime string "nodate" at position 0'

    def test_accepts_flat_format(self):
        config = dict(num=1, date=['2000-01-01', '2000-01-02'], location=[5, 9])
        makrel.load_config(config)

    def test_accepts_group_format(self):
        config = dict(groups=[
            dict(num=1, date=['2000-01-01', '2000-01-02'], location=[5, 9])
        ])
        makrel.load_config(config)

    def test_accepts_list_format(self):
        config = [
            dict(num=1, date=['2000-01-01', '2000-01-02'], location=[5, 9])
        ]
        makrel.load_config(config)


class Test_point_inside_polygon:
    def test_inside_if_clockwise_square(self):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        x, y = makrel.point_inside_polygon(coords)
        assert 0 < x < 1
        assert 0 < y < 1

    def test_inside_if_counterclockwise_square(self):
        coords = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        x, y = makrel.point_inside_polygon(coords)
        assert 0 < x < 1
        assert 0 < y < 1


class Test_get_polygons_from_feature_geometry:
    def test_correct_number_of_polygons_if_multipolygon(self):
        feature_geom = {
            "type": "MultiPolygon",
            "coordinates": [
                [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
                [[[10, 10], [11, 10], [11, 11], [10, 10]]],
                [[[20, 10], [21, 10], [21, 11], [20, 10]]],
            ],
        }
        p = makrel.get_polygons_from_feature_geometry(feature_geom)
        assert len(p) == 3

    def test_cuts_last_coordinate_if_duplicated(self):
        feature_geom = {
            "type": "MultiPolygon",
            "coordinates": [[[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]],
        }
        p = makrel.get_polygons_from_feature_geometry(feature_geom)
        assert len(p[0]) == 4

    def test_polygons_and_multipolygons_give_same_output(self):
        crd = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]
        feature_geom_poly = {"type": "Polygon", "coordinates": [crd]}
        feature_geom_multi = {"type": "MultiPolygon", "coordinates": [[crd]]}
        poly = makrel.get_polygons_from_feature_geometry(feature_geom_poly)
        multi = makrel.get_polygons_from_feature_geometry(feature_geom_multi)
        assert np.shape(poly) == np.shape(multi)
