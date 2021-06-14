import datetime

time_reversal = True

if time_reversal:
    release_file = "reverse.rls"
    start = datetime.datetime(2015, 4, 7)
    sgn = -1
else:
    release_file = "salmon_lice.rls"
    start = datetime.datetime(2015, 3, 28)
    sgn = 1


mult = 1
release_period = datetime.timedelta(days=3)
nrelease = 5  # Number of releases

X1, Y1, farmid1, super1 = 379, 539, 10333, 72345.0
X2, Y2, farmid2, super2 = 381, 523, 10444, 32028.0


with open(release_file, mode="w") as f:
    f.write("mult         release_time      X      Y     Z   farmid   super\n")
    for i in range(nrelease):
        time_ = start + i * sgn* release_period
        f.write(
            f"{mult:4d}  {time_.isoformat()} {X1:6.1f} {Y1:6.1f} "
            f"  5.0   {farmid1:5d} {super1:7.1f}\n"
        )
        f.write(
            f"{mult:4d}  {time_.isoformat()} {X2:6.1f} {Y2:6.1f} "
            f"  5.0   {farmid2:5d} {super2:7.1f}\n"
        )
