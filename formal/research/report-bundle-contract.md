# Recoverable verified report bundle v2

Registration: `research-notes/registrations/report-bundle-engineering.json`.
Existing export admission, reconciliation, report bytes and default directory
export remain unchanged. New CLI flag `--report-bundle-v2` explicitly requests
one versioned bundle in an existing output directory outside the source archive.
The publisher defaults disabled, accepts exactly the seven report names, encodes
UTF-8 report contents with fixed research-only metadata, and never trains or
promotes a policy. No historical archive or holdout is opened in this work.

Maximum individual report4MiB, combined report16MiB, encoded bundle32MiB.
Admission and complete encoding precede filesystem effects. The trusted caller
supplies the intended already-existing directory. A directory descriptor binds
all relative leaf operations. Target name is fixed `report-bundle-v2.json`.
Private staging names use fresh128-bit random suffixes and exclusive creation;
collision produces failure, never truncation. Staging receives complete bytes,
then file fsync. An atomic exclusive hard link publishes the target only after
that fsync. Parent-directory fsync precedes success acknowledgement.

Retry: an existing regular target must compare byte-for-byte with the newly
verified complete bundle; mismatch, symlink or malformed target fails closed.
A matching target is fsynced and the directory is fsynced before acknowledging
success. Concurrent identical writers may share the same result; differing
writers cannot replace the winner. No existing file is opened for writing,
truncated or unlinked. Only this call's successfully created staging leaf may
be unlinked by its cleanup; failed cleanup or process death may leave an orphan.
Orphans are never treated as completed evidence or resumed as writable files.
No automatic orphan scavenging is introduced. Adequate free space and finitely
many crashes are explicit progress assumptions.

Publication has three states: absent, visible but not directory-durable, and
acknowledged durable. A crash before acknowledgement may lose the target name
or retain a complete durable file; retry handles either case. After success,
the full fixed bundle survives modeled crashes. The abstraction trusts exclusive
open/link and fsync semantics of a local POSIX filesystem and device that honors
flushes. It excludes hostile namespace/content mutation, remote/object filesystems,
OS/compiler defects, media failure and forged verified source evidence. Actual
fault tests kill processes, not power to hardware. No universal storage theorem.

F-RL-BUNDLE-SOURCE: complete source, exporter admission-to-publisher composition,
default-disabled version gate and effect inventory. F-RL-BUNDLE-ARITH: bounded
partial-write cursor and byte-identity publication conditions (SMT).
F-RL-BUNDLE-FLOW: finite two-writer crash/retry/interleaving model proving no partial
publication, no overwrite, matching acknowledgement and conditional recovery.
F-RL-BUNDLE-CONFORMANCE: deterministic actual filesystem failures, process crashes,
concurrent writers, byte parity with legacy reports and idempotent retry.

Original38 titles/scope/closure criteria are preserved. Obligation37 gains partial
composed durable report-export evidence; training-run archives and inherited bot
snapshot recovery remain unverified, so broad recovery is not closed. No other
broad closure is inferred. Champion preservation must be re-established for both
old exclusive writers and new exclusive-link publication before merge.
