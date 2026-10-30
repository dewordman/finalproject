import mailbox, hashlib

def keys(path):
    out = set()
    for m in mailbox.mbox(path):
        try:
            body = m.get_payload(decode=False)
            body = body if isinstance(body, str) else str(body)
            out.add(hashlib.sha256((str(m["Subject"]) + body).encode("utf-8", "replace")).hexdigest())
        except Exception:
            pass
    return out

a = keys(r"data\raw\nazario\20051114.mbox")
b = keys(r"data\raw\nazario\phishing1.mbox")
print("20051114:", len(a), "| phishing1:", len(b), "| shared:", len(a & b))