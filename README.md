# auto-coding

## Execution timeouts

New schema-v2 plans should explicitly propose `implementation_seconds: 1800`
(30 minutes) and `review_seconds: 600` (10 minutes). These are recommended
authoring values, not runtime substitutions: both fields and `test_seconds` are
mandatory and must be approved before freezing. There is no plan generator or
implicit implementation/review default. Existing frozen plans remain unchanged.

`test_seconds` applies to each approved command. Use short limits for short
tests; 1800 seconds is supported. The existing validation maximum remains
86400 seconds to preserve compatibility with already approved plans.

`plan preflight` and `run` default to 120 seconds per Git command, including
scope/evidence queries. `--git-timeout-seconds 10` explicitly overrides that
default. There are no automatic extensions or retries.

Confirmed timeouts print a sanitized `AUTO-CODING STOPPED - TIMEOUT` diagnostic.
Stopped checkpoints may include an optional `timeout` record (operation,
configured seconds, observed elapsed seconds or null, and `TIMEOUT`
classification). Legacy v1/v2 checkpoints remain readable without that field.
JSON and Markdown reports include it with the batch/task identity, stop state
and publication history. Preflight failure before checkpoint creation produces
a terminal diagnostic and result only; it does not create execution metadata.
Unavailable Git elapsed time remains null. Process elapsed times include cleanup.

Timeouts preserve ambiguous Git outcomes and never authorize publication or
later tasks. Windows cleanup uses the existing bounded process-tree termination;
detached descendants and denied `taskkill` permissions cannot be guaranteed
contained. Git retains its existing subprocess timeout behavior; a timed-out
commit or push may have completed and requires manual investigation.

## Manual continuation of a stopped sequential batch

Continuation creates a separate run. It never resumes the stopped batch, copies
its dirty working files, edits its frozen plan, or overwrites its metadata.

First inspect using the **original** approval supplied through your trusted
approval channel (a fingerprint is not authenticated human identity):

```powershell
py -m auto_coding continuation inspect `
  --source-plan "C:\Metadata\d24\frozen.json" `
  --source-approved-sha256 $originalApprovedSha `
  --source-metadata "C:\Metadata\d24" `
  --repository "C:\Projects\Trail Atlas"
```

Inspection validates the stopped schema-v2 ledger and both reports against the
original frozen plan, cross-checks child checkpoints and published Git commit
parents/scope/staging hashes, and verifies the source branch/origin/local HEAD
and current remote SHA. Dirty/staged failed-task files stay in the original
repository. Ambiguous publication, incomplete observations, uncertain process
cleanup, unfinalized publication metadata or changed remote state stop preparation.
If no task was published, inspection can identify the verified original baseline.
Metadata is not cryptographic proof of execution; checks do not lock Git state.

Author a **new schema-v2 proposed plan**, with a different batch ID, a new feature
branch, the reported base SHA and the same intended origin. Include only the
remaining/revised work, renumber dependencies as appropriate, and explicitly
review all prompts, models, scopes, criteria, commands and timeouts. Preparation
does not edit tasks or infer new instructions. Reusing a published task ID is
rejected; human review must detect semantically repeated work under different IDs.

```powershell
py -m auto_coding continuation prepare `
  --source-plan "C:\Metadata\d24\frozen.json" `
  --source-approved-sha256 $originalApprovedSha `
  --source-metadata "C:\Metadata\d24" `
  --repository "C:\Projects\Trail Atlas" `
  --plan "C:\Metadata\d24b-proposed.json" `
  --destination "C:\Projects\Trail Atlas D24b" `
  --metadata-directory "C:\Metadata\d24b" `
  --dry-run
```

Remove `--dry-run` to create the fresh clone, local branch and metadata directory.
Both destinations must be nonexistent, with existing parents, outside the original
repository/metadata and disjoint from one another; symlinks/junctions are rejected.
Only committed remote objects are cloned. Preparation never commits or pushes.
It preserves partial new directories after failure rather than deleting or retrying.

For example, **if inspection verifies it**, a D24b plan can propose
`expected_head: 5b42be326b787796bd992500968e986c72e16493` and
`expected_branch: feature/d24b-trail-atlas`. The real D24 repository is not touched
by automated tests.

From the new clone, explicitly publish its new branch before normal preflight;
the existing gates require that remote branch to exist at the approved base:

```powershell
git -C "C:\Projects\Trail Atlas D24b" push origin feature/d24b-trail-atlas
py -m auto_coding plan validate "C:\Metadata\d24b-proposed.json"
# Review the COMPLETE new plan and independently confirm its new fingerprint.
py -m auto_coding plan freeze "C:\Metadata\d24b-proposed.json" `
  --approved-sha256 $newApprovedSha --destination "C:\Metadata\d24b\frozen.json" `
  --repository "C:\Projects\Trail Atlas D24b"
py -m auto_coding plan preflight "C:\Metadata\d24b\frozen.json" `
  --approved-sha256 $newApprovedSha --repository "C:\Projects\Trail Atlas D24b"
py -m auto_coding run "C:\Metadata\d24b\frozen.json" `
  --approved-sha256 $newApprovedSha --repository "C:\Projects\Trail Atlas D24b" `
  --metadata-directory "C:\Metadata\d24b" --repository-id "example/trail-atlas" `
  --evidence-max-bytes 1048576 --packet-max-bytes 2097152 --report-max-bytes 2097152
```

Git identity and remote access must be configured normally for the fresh clone;
credentials and local configuration are not copied from the dirty repository.
`continuation.json` retains source batch/plan/checkpoint/report fingerprints,
published task IDs, the base SHA and the new batch/plan/destination identities.
Run validates this optional reference against its separately approved new plan
before invoking Codex. It remains a provenance reference, not approval or a
signed execution attestation. Changed proposed plans need a fresh preparation.
Existing checkpoint/report schemas and the normal single/multi-task pipeline
remain unchanged. Incomplete original audit records require manual investigation.
