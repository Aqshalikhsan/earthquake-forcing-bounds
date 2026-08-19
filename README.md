# Site branch

The published site for
[earthquake-forcing-bounds](https://github.com/Aqshalikhsan/earthquake-forcing-bounds),
served at
[earthquake-forcing-bounds.vercel.app](https://earthquake-forcing-bounds.vercel.app).

This is an orphan branch. It shares no history with `main` and carries none of
the data archives, so a deployment clones under 4 MB rather than the whole
repository. The analysis code and data live on `main`.

| File | Contents |
| --- | --- |
| `index.html` | what the thirteen candidate forcings return, and the two findings that matter more than any individual bound |
| `audit.html` | the audit tool, with what it can and cannot do stated on the page |
| `fingerprint.js` | fifteen perturbation axes, ported from `src/audit/fingerprint.py` on `main` |
| `audit.js` | tree evaluation, conformal prediction sets, abstention outside the calibrated range |
| `audit_model.json` | 1400 gradient-boosted trees exported node by node |

Everything runs in the browser. Nothing is uploaded, because nothing needs to
be: the fingerprint is cheap arithmetic and the model is small.

Two checks guard the port from Python, and both fail the build rather than
shipping a silent disagreement. The exported ensemble is re-evaluated with an
independent implementation and matches scikit-learn exactly. The fifteen axes
are recomputed here and match the Python originals to 2.7e-11 in the worst
case.

## Editing

Source lives in `site/` on `main` and is copied here. To change the site, edit
there, then update this branch from that directory.
