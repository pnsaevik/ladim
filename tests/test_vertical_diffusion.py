import numpy as np
import pytest

from ladim.vertical_diffusion import (
    ConstantDiffusor, LaBolleDiffusor, VerticalDiffusor, _reflected_step,
)

INF = float("inf")


def create_constant(value):
    return VerticalDiffusor.create(dict(method="constant", value=value))


def create_labolle(min=0, max=INF, resolution=INF):
    return VerticalDiffusor.create(
        dict(method="labolle", min=min, max=max, resolution=resolution))


def particles(n, z0, depth=1000.0):
    return (np.zeros(n), np.zeros(n), np.full(n, float(z0)),
            np.full(n, float(depth)), np.arange(n, dtype=np.int64))


def uniform_mix(value):
    return lambda x, y, z: np.full(len(z), value)


# ---------- Factory ----------

class Test_create:
    def test_creates_diffusor_of_given_method(self):
        assert isinstance(create_constant(0.01), ConstantDiffusor)
        assert isinstance(create_labolle(), LaBolleDiffusor)

    def test_number_is_constant_diffusivity(self):
        for spec in (0.01, dict(method="constant", value=0.01)):
            d = VerticalDiffusor.create(spec)
            assert isinstance(d, ConstantDiffusor) and d.value == 0.01
        assert VerticalDiffusor.create(2).value == 2.0

    def test_no_diffusivity_is_disabled(self):
        for spec in (None, 0, 0.0, dict(method="constant", value=0)):
            assert VerticalDiffusor.create(spec).enabled is False

    def test_labolle_string_has_no_clipping_or_binning(self):
        d = VerticalDiffusor.create("labolle")
        assert isinstance(d, LaBolleDiffusor)
        assert (d.min, d.max, d.resolution) == (0, INF, INF)

    def test_does_not_modify_specification(self):
        spec = dict(method="labolle", min=0, max=1, resolution=5)
        VerticalDiffusor.create(spec)
        assert spec == dict(method="labolle", min=0, max=1, resolution=5)

    def test_stores_fields(self):
        d = create_labolle(min=1e-4, max=1e-2, resolution=5)
        assert (d.min, d.max, d.resolution) == (1e-4, 1e-2, 5.0)
        assert create_constant(2).value == 2.0

    def test_accepts_infinity_as_string(self):
        d = create_labolle(min=0, max="inf", resolution="Infinity")
        assert (d.max, d.resolution) == (INF, INF)

    def test_diffusivity_function_is_not_a_field(self):
        # The function is given to diffuse(), not to the factory
        with pytest.raises(ValueError):
            VerticalDiffusor.create(dict(method="constant", value=1, vert_mix=len))
        with pytest.raises(ValueError):
            VerticalDiffusor.create(
                dict(method="labolle", min=0, max=1, resolution=5, vert_mix=len))

    @pytest.mark.parametrize("spec", [
        "labolle_5",  # Unknown string
        "constant",  # Fields are missing
        True,
        [1.0],
        dict(value=1.0),  # No method
        dict(method="foo", value=1.0),
        dict(method=None),
        dict(method="constant"),  # No default values
        dict(method="constant", value=1.0, min=0),  # Unknown field
        dict(method="labolle", min=0, max=1),  # No resolution
        -1.0,
        INF,
        dict(method="constant", value=True),
        dict(method="constant", value="a"),
        dict(method="constant", value=[1]),
        dict(method="labolle", min=2, max=1, resolution=5),
        dict(method="labolle", min=-1, max=1, resolution=5),
        dict(method="labolle", min=0, max=1, resolution=0),
    ])
    def test_rejects_invalid_specification(self, spec):
        with pytest.raises(ValueError):
            VerticalDiffusor.create(spec)

    def test_unknown_method_lists_available_methods(self):
        with pytest.raises(ValueError, match="constant, labolle"):
            VerticalDiffusor.create(dict(method="foo"))
        with pytest.raises(ValueError, match="constant, labolle"):
            VerticalDiffusor.create("foo")


# ---------- Constant diffusivity ----------

class Test_constant:
    def test_zero_diffusivity_is_disabled_and_does_not_move_particles(self):
        d = create_constant(0)
        assert d.enabled is False
        X, Y, Z, depth, pid = particles(10, 5.0)
        assert np.array_equal(d.diffuse(X, Y, Z, depth, pid, 600), Z)

    def test_returns_new_array(self):
        X, Y, Z, depth, pid = particles(10, 5.0)
        Z1 = create_constant(0.01).diffuse(X, Y, Z, depth, pid, 600)
        assert Z1 is not Z and np.all(Z == 5.0)

    def test_has_correct_variance(self):
        k, dt, n = 0.01, 600, 200_000
        X, Y, Z, depth, pid = particles(n, 500.0)
        np.random.seed(1)
        dz = create_constant(k).diffuse(X, Y, Z, depth, pid, dt) - 500.0
        assert dz.mean() == pytest.approx(0, abs=0.01)
        assert dz.var() == pytest.approx(2 * k * dt, rel=0.02)
        assert np.all(np.abs(dz) <= (6 * k * dt) ** 0.5 + 1e-4)  # Uniform noise

    @pytest.mark.parametrize("k", [0.01, 1.0, 100.0])  # Last: steps >> depth
    def test_stays_between_surface_and_bottom(self, k):
        depth = 20.0
        X, Y, Z, d, pid = particles(50_000, 0.0, depth)
        diffusor = create_constant(k)
        np.random.seed(2)
        for _ in range(30):
            Z = diffusor.diffuse(X, Y, Z, d, pid, 600)
            assert Z.min() >= 0 and Z.max() <= depth

        if k >= 1.0:  # Well mixed: uniform distribution
            assert Z.mean() == pytest.approx(depth / 2, rel=0.02)
            assert Z.var() == pytest.approx(depth ** 2 / 12, rel=0.03)

    def test_reflects_at_surface_and_bottom(self):
        # With zero diffusivity, only the folding of the positions is exercised
        Z0 = np.array([-3.0, 13.0, 25.0, -27.0, 0.0, 10.0, 20.0, 3.0])
        depth = np.array([10.0] * 7 + [np.nan])
        Z = np.empty(len(Z0))
        _reflected_step(Z0, np.zeros(len(Z0)), np.ones(len(Z0)), depth, Z)
        assert Z.tolist() == [3.0, 7.0, 5.0, 7.0, 0.0, 10.0, 0.0, 3.0]  # Last: no depth

    def test_is_reproducible(self):
        def run(seed):
            X, Y, Z, depth, pid = particles(1000, 5.0, 50.0)
            np.random.seed(seed)
            return create_constant(0.1).diffuse(X, Y, Z, depth, pid, 600)

        assert np.array_equal(run(3), run(3))
        assert not np.allclose(run(3), run(4))

    def test_follows_particle_identifier(self):
        X, Y, Z, depth, pid = particles(1000, 5.0, 50.0)
        np.random.seed(3)
        a = create_constant(0.1).diffuse(X, Y, Z, depth, pid, 600)
        np.random.seed(3)
        b = create_constant(0.1).diffuse(X, Y, Z, depth, pid[::-1].copy(), 600)
        assert np.array_equal(a, b[::-1])


# ---------- Diffusivity from the ocean model (labolle) ----------

class Test_labolle:
    def test_constant_diffusivity_has_correct_variance(self):
        k, dt, n = 0.01, 600, 200_000
        X, Y, Z, depth, pid = particles(n, 500.0)
        np.random.seed(1)
        d = create_labolle()
        dz = d.diffuse(X, Y, Z, depth, pid, dt, uniform_mix(k)) - 500.0
        assert dz.mean() == pytest.approx(0, abs=0.01)
        assert dz.var() == pytest.approx(2 * k * dt, rel=0.02)

    def test_requires_diffusivity_function(self):
        X, Y, Z, depth, pid = particles(10, 5.0)
        with pytest.raises(ValueError, match="vert_mix"):
            create_labolle().diffuse(X, Y, Z, depth, pid, 600)
        with pytest.raises(ValueError, match="vert_mix"):
            create_labolle().diffuse(X, Y, Z, depth, pid, 600, vert_mix=0.1)

    def test_constant_method_does_not_use_diffusivity_function(self):
        def vert_mix(x, y, z):
            raise AssertionError("Should not be called")

        X, Y, Z, depth, pid = particles(10, 5.0)
        np.random.seed(1)
        Z1 = create_constant(0.01).diffuse(X, Y, Z, depth, pid, 600, vert_mix)
        assert np.all(Z1 != 5.0)

    def test_passes_positions_and_depths_to_diffusivity_function(self):
        calls = []

        def vert_mix(x, y, z):
            calls.append((x.copy(), y.copy(), z.copy()))
            return np.full(len(z), 1e-3)

        X, Y, Z, depth, pid = particles(10, 7.3)
        X = np.arange(10.0)
        Y = np.arange(10.0) * 2
        np.random.seed(1)
        create_labolle().diffuse(X, Y, Z, depth, pid, 600, vert_mix)

        assert len(calls) == 2  # Initial and intermediate position
        for x, y, z in calls:
            assert np.array_equal(x, X) and np.array_equal(y, Y)
        assert np.all(calls[0][2] == 7.3)  # No binning
        assert np.all(calls[1][2] != 7.3)

    def test_samples_at_bin_midpoints(self):
        calls = []

        def vert_mix(x, y, z):
            calls.append(z.copy())
            return np.full(len(z), 1e-3)

        n = 5000
        X, Y, _, depth, pid = particles(n, 0.0)
        Z = np.random.default_rng(0).uniform(0, 50, n)
        np.random.seed(1)
        create_labolle(resolution=5).diffuse(X, Y, Z, depth, pid, 600, vert_mix)

        assert np.array_equal(calls[0], (Z // 5) * 5 + 2.5)
        for z in calls:
            assert np.all(z % 5 == 2.5)
        assert calls[0].min() == 2.5  # The top bin is sampled at 2.5 m

    def test_final_step_uses_diffusivity_at_intermediate_position(self):
        # Strong mixing above 10 m, tiny below. Particles starting in the weak
        # layer which step into the strong layer get a large final step
        dt, z0 = 600, 10.0 + 1e-3
        X, Y, Z, depth, pid = particles(20_000, z0)
        np.random.seed(2)
        vert_mix = lambda x, y, z: np.where(z < 10, 1.0, 1e-6)
        dz = create_labolle().diffuse(X, Y, Z, depth, pid, dt, vert_mix) - z0
        tiny = (6 * 1e-6 * dt) ** 0.5  # Largest step with the weak diffusivity
        assert np.any(np.abs(dz) > 100 * tiny)
        assert np.any(np.abs(dz) <= tiny)

    def test_clips_diffusivity(self):
        dt = 600
        for k, expected in [(1.0, 1e-2), (1e-9, 1e-4)]:
            X, Y, Z, depth, pid = particles(100_000, 500.0)
            np.random.seed(1)
            d = create_labolle(min=1e-4, max=1e-2, resolution=5)
            dz = d.diffuse(X, Y, Z, depth, pid, dt, uniform_mix(k)) - 500.0
            assert dz.var() == pytest.approx(2 * expected * dt, rel=0.03)

    def test_replaces_invalid_diffusivity_with_minimum(self):
        X, Y, Z, depth, pid = particles(100, 5.0)
        np.random.seed(1)
        d = create_labolle(min=0, max=1, resolution=5)
        assert np.all(d.diffuse(X, Y, Z, depth, pid, 600, uniform_mix(np.nan)) == 5.0)

        d = create_labolle(min=1e-2, max=1, resolution=5)
        assert np.all(d.diffuse(X, Y, Z, depth, pid, 600, uniform_mix(np.nan)) != 5.0)

    def test_stays_between_surface_and_bottom(self):
        depth = 20.0
        X, Y, Z, d, pid = particles(20_000, 0.0, depth)
        diffusor = create_labolle(resolution=5)
        np.random.seed(2)
        for _ in range(20):
            Z = diffusor.diffuse(X, Y, Z, d, pid, 600, uniform_mix(1.0))
            assert Z.min() >= 0 and Z.max() <= depth

    def test_is_reproducible(self):
        def run(seed):
            X, Y, Z, depth, pid = particles(1000, 5.0, 50.0)
            np.random.seed(seed)
            return create_labolle().diffuse(X, Y, Z, depth, pid, 600, uniform_mix(0.1))

        assert np.array_equal(run(3), run(3))
        assert not np.allclose(run(3), run(4))

    def test_raises_error_from_diffusivity_function(self):
        def vert_mix(x, y, z):
            raise KeyError("The forcing files contain neither AKs nor ln_AKs")

        X, Y, Z, depth, pid = particles(10, 5.0)
        with pytest.raises(KeyError, match="AKs"):
            create_labolle().diffuse(X, Y, Z, depth, pid, 600, vert_mix)
