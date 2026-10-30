
import mailbox, glob, email.utils, collections

for folder in (r"data\raw\nazario", r"data\raw\nazario_external"):
    print("==", folder)
    for p in sorted(glob.glob(folder + r"\*.mbox")):
        with open(p, "rb") as f:
            start = f.read(5)
        n = 0
        years = collections.Counter()
        for m in mailbox.mbox(p):
            n += 1
            try:
                years[email.utils.parsedate_to_datetime(m["Date"]).year] += 1
            except Exception:
                years["no_date"] += 1
        print(p, "| starts with", start, "|", n, "messages |", dict(years))


