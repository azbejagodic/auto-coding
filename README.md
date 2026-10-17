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
