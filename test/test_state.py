import numpy as np  # type: ignore
import pytest

from ladim2.state import State

# ------------
# __init__
# ------------


def test_init_minimal():
    """Init State with no arguments"""
    S = State()
    assert len(S) == 0
    assert S.npid == 0
    assert set(S.variables) == {"pid", "X", "Y", "Z", "active", "alive"}
    assert S.instance_variables == {"pid", "X", "Y", "Z", "active", "alive"}
    assert S.particle_variables == set()
    assert S.pid.dtype == int
    assert all(S.pid == [])
    assert S.X.dtype == np.float64
    assert all(S.variables["X"] == [])
    assert all(S["X"] == [])
    assert all(S.X == [])
    assert S.Y.dtype == float
    assert S.Z.dtype == "f8"
    assert S.alive.dtype == bool
    assert S.default_values["alive"]


def test_init_args():
    """Init State with extra variables"""
    S = State(
        instance_variables=dict(age=float, stage=int),
        particle_variables=dict(release_time="time"),
        default_values=dict(age=0, stage=1),
    )
    assert "age" in S.instance_variables
    assert S.age.dtype == float
    assert S.default_values["age"] == 0
    assert S.stage.dtype == int
    assert S.particle_variables == {"release_time"}
    assert S.release_time.dtype == np.dtype("M8[s]")
    assert S.dtypes["release_time"] == np.dtype("M8[s]")
    assert all(S.release_time == np.array([], np.datetime64))


def test_override_mandatory():
    S = State(instance_variables=dict(X="f4"))
    assert S.X.dtype == np.float32


def test_set_default_err1():
    """Trying to set default for an undefined variable"""
    with pytest.raises(ValueError):
        S = State(particle_variables={"age": float}, default_values=dict(length=4.3))


def test_set_default_err2():
    """Trying to set default for pid"""
    with pytest.raises(ValueError):
        S = State(default_values=dict(pid=42))


def test_set_default_err3():
    """Trying to set an array as default value"""
    with pytest.raises(TypeError):
        S = State(
            instance_variables=dict(length=float),
            default_values=dict(length=[1.2, 4.3]),
        )


# --------------------
# append
# --------------------


def test_append_scalar():
    state = State()
    state.append(X=200, Z=5, Y=100)
    assert len(state) == 1
    assert state.npid == 1
    assert np.all(state.pid == [0])
    assert np.all(state.active == [True])
    assert np.all(state.alive == [True])
    assert np.all(state.X == [200])


def test_append_array():
    """Append an array to a non-empty state"""
    state = State()
    state.append(X=200, Z=5, Y=100)
    length = len(state)
    npid = state.npid
    state.append(X=np.array([201, 202]), Y=110, Z=[5, 10])
    assert len(state) == length + 2
    assert state.npid == npid + 2
    assert np.all(state.pid == [0, 1, 2])
    assert np.all(state["pid"] == [0, 1, 2])
    assert np.all(state.variables["pid"] == [0, 1, 2])
    assert np.all(state.active == 3 * [True])
    assert np.all(state.alive == 3 * [True])
    assert np.all(state.X == [200, 201.0, 202.0])
    assert np.all(state["X"] == [200, 201.0, 202.0])
    assert np.all(state.variables["X"] == [200, 201.0, 202.0])
    assert np.all(state.Y == [100.0, 110.0, 110.0])
    assert np.all(state.Z == [5.0, 5.0, 10.0])


def test_extra_instance_variables():
    """Append with extra instance variables, with and without default"""
    state = State(
        instance_variables=dict(age=float, stage="int"), default_values=dict(stage=1)
    )
    assert len(state) == 0
    assert state.age.dtype == float
    assert state.stage.dtype == int
    state.append(X=[1, 2], Y=2, Z=3, age=0)
    assert len(state) == 2
    assert all(state.age == [0, 0])
    assert all(state.stage == [1, 1])


def test_append_nonvariable():
    """Append an undefined variable"""
    state = State()
    with pytest.raises(ValueError):
        state.append(X=1, Y=2, Z=3, length=20)


def test_append_missing_variable():
    state = State()
    # with pytest.raises(TypeError):
    # Now Y becomes NaN, correct behaviuour??
    state.append(X=100, Z=10)
    assert state.Y[0] != state.Y[0]


def test_append_missing_particle_variable():
    state = State(particle_variables=dict(X_start=float))
    # with pytest.raises(TypeError):
    state.append(X=100, Y=200, Z=5)
    assert state.X_start[0] != state.X_start[0]


def test_append_shape_mismatch():
    state = State()
    with pytest.raises(ValueError):
        state.append(X=[100, 101], Y=[200, 201, 202], Z=5)


def test_missing_default():
    state = State(
        instance_variables=dict(age=float, stage=int), default_values=dict(age=0)
    )
    # No default for stage
    # with pytest.raises(TypeError):
    #    state.append(X=1, Y=2, Z=3)
    # changed behaviour: now check for NaN
    state.append(X=1, Y=2, Z=3)
    assert state.stage[0] != state.stage[0]


def test_not_append_pid():
    """Can not append to pid"""
    S = State()
    with pytest.raises(ValueError):
        S.append(X=10, Y=20, Z=5, pid=101)


# ----------------
# Update
# ----------------


def test_variable_update():
    """Update a variable, low level"""
    S = State()
    S.append(X=[100, 110], Y=[200, 210], Z=5)
    S.variables["X"] += 1
    assert all(S.variables["X"] == [101, 111])


def test_update_item():
    """Item style variable update is OK"""
    S = State()
    S.append(X=[100, 110], Y=[200, 210], Z=5)
    S["X"] += 1
    assert all(S.variables["X"] == [101, 111])


def test_update_attr():
    """Attribute style assignment to variables is not allowed"""
    S = State()
    S.append(X=[100, 110], Y=[200, 210], Z=5)
    with pytest.raises(AttributeError):
        S.X += 1


def test_update_error_not_variable():
    S = State()
    S.append(X=[100, 110], Y=[200, 210], Z=5)
    with pytest.raises(KeyError):
        S["Lon"] = [4.5, 4.6]


def test_update_error_wrong_size():
    # Alternative broadcast the scalar, equivalent to s["X"] = [110, 100]
    S = State()
    S.append(X=[100, 110], Y=[200, 210], Z=5)
    with pytest.raises(KeyError):
        S["X"] = 110
    with pytest.raises(KeyError):
        S["X"] = [101, 111, 121]


# --------------
# Compactify
# --------------


def test_compactify():
    S = State(default_values=dict(Z=5))
    S.append(X=[10, 11], Y=[1, 2])
    assert len(S) == 2
    S.append(X=[21, 22], Y=[3, 4])
    assert len(S) == 4
    # Kill second particle
    S.alive[1] = False
    S.compactify()
    assert len(S) == 3
    assert S.npid == 4
    assert np.all(S.active)
    assert np.all(S.alive)
    assert np.all(S.pid == [0, 2, 3])
    assert np.all(S.X == [10, 21, 22])
    # The arrays should be contiguous after removing an element
    assert S.X.flags["C_CONTIGUOUS"]


def test_not_compactify_particle_variables():
    S = State(particle_variables=dict(X0=float), default_values=dict(Z=5))
    X0 = [10, 11, 12, 13]
    Y0 = [20, 21, 22, 23]
    S.append(X=X0, Y=Y0, X0=X0)
    S.alive[1] = False
    S.compactify()
    assert len(S) == 3
    assert all(S.pid == [0, 2, 3])
    assert all(S.X == [10, 12, 13])
    # particle_variable X0 is not compactified
    assert all(S.X0 == X0)


def test_update_and_append_and_compactify():
    """Check that updating bug has been fixed"""
    S = State()

    # One particle
    S.append(X=100, Y=10, Z=5)
    assert all(S.pid == [0])
    assert all(S.X == [100])

    # Update position
    S["X"] += 1
    assert all(S.pid == [0])
    assert all(S.X == [101])

    # Update first particle and add two new particles
    S["X"] += 1
    S.append(X=np.array([200, 300]), Y=np.array([20, 30]), Z=5)
    assert all(S.pid == [0, 1, 2])
    assert all(S.X == [102, 200, 300])

    # Update particle positions and kill the first particle, pid=0
    S["X"] = S["X"] + 1.0
    S["alive"][0] = False
    S.compactify()
    assert all(S.pid == [1, 2])
    assert all(S.X == [201, 301])

    # Update positions
    S["X"] = S["X"] + 1
    assert all(S.pid == [1, 2])
    assert all(S.X == [202, 302])
