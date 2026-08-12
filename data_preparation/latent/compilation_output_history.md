# Recoverable latent compilation output history

The compiler's stdout and stderr were attached to anonymous write-only pipes.
Earlier progress-bar output was not persisted by the process and cannot be
recovered after it was emitted. This file records all durable compiler-log
entries and status observations recovered during this run.

## Durable compiler log entries

```text
[1/5] LxYxvv/quora_qa_raw: SUCCESS (370_840 examples)

[1/3] recursal/Fanatic-Fandom: SUCCESS (138_616 examples)
[2/3] PleIAs/SYNTH: FAIL during trajectifying: ArrowInvalid: offset overflow
while concatenating arrays, consider casting input from `string` to
`large_string` first.
```

## Fix and retry

The overflow was fixed in `trajectory.py` by using bounded `_take_chunked`
gathers for metadata columns and their episode comparisons. The retry started
with `PleIAs/SYNTH`, followed by `Lyun0912/LongABC`.

At 2026-08-11T23:00:14Z, SYNTH was still active (PID 1940062), transferring
approximately 283 MB every 10 seconds over HTTPS. No `PleIAs--SYNTH` files
had yet appeared in the Hub repository, so the upload had not reached its
final visible commit stage.
