#!/usr/bin/env bash
# Extracted from queue-retry-handler.yml's "Retry once" step (a proven fix from this fleet's
# own rollout): GitHub hard-caps a run: block containing ANY ${{ }} interpolation at ~21,000
# characters, evaluated as a single expression -- confirmed live elsewhere in this org's own
# rollout (0 real workflow_run-triggered executions, every attempt instead showing event:push,
# zero jobs, and name falling back to the file path -- the signature of a workflow GitHub
# couldn't parse/start at all). This step's own script
# already carries zero inline ${{ }} (every dynamic value comes through this step's own env:
# block, unaffected by the cap), but crossed 21,000 characters anyway as this round's fix grew
# it further -- moved to a checked-out file so the run: block itself (what's measured) stays a
# single invocation line regardless of how large this script grows from future review rounds'
# comments.
#
# set -eo pipefail, not the caller's own bash -e alone: invoking this file as a subprocess
# does NOT inherit the run: step's shell options -- GitHub Actions' default `bash -eo
# pipefail {0}` applies to the OUTER command (running this script), not to commands INSIDE it.
# Every unguarded (non `|| true`) command below relied on that default when this logic lived
# inline; declaring it explicitly here preserves that behavior exactly.
set -eo pipefail

# Env vars expected (set via the calling step's own env: block, inherited by this subprocess):
#   GH_TOKEN, READ_TOKEN, PR, LABEL, HEAD_SHA12, HEAD_SHA, GITHUB_REPOSITORY (implicit) or REPO
REPO="${REPO:-$GITHUB_REPOSITORY}"

# Shared by both fail-closed branches below: a bare needs-human POST with `|| true`
# would let this whole step report success even when the ONE signal meant to get a
# human's attention never landed. Verify it, and propagate failure via this
# function's own exit status so each call site can fail the step -- mark-handled's
# gate on `steps.retry.conclusion == 'success'` then correctly withholds the dedup
# marker, leaving the attempt eligible for a rerun to actually deliver the signal.
attach_needs_human_or_fail() {
  # Recheck the head immediately before pausing (Codex P1, round 8, propagation migration):
  # both callers of this shared helper already checked the head at THEIR OWN call site
  # (before deciding to call this at all), but time passes between that check and this
  # function's own label POST below (an earlier gh label create attempt, prior retries in the
  # calling branch, etc.) -- a replacement commit landing in that window would get a PR-wide
  # pause and an old-HEAD_SHA12 provenance marker attached to it instead of the commit this
  # call was actually evaluating. Especially persistent for a Dependabot-authored PR here:
  # dependabot-auto-merge.yml's own admission step only DISARMS when it observes needs-human
  # (it has no staleness-clearing logic of its own), and auto-merge-trigger.yml now excludes
  # the dependabot[bot] actor entirely (see that file's own header) -- so nothing would ever
  # clear a wrongly-attached pause on a Dependabot PR's new revision, unlike every other
  # migrated repo where auto-merge-trigger.yml's needs_human_is_stale() eventually catches it.
  local current_head_sha_final
  current_head_sha_final=$(GH_TOKEN="$READ_TOKEN" gh pr view "$PR" --repo "$REPO" --json headRefOid --jq .headRefOid 2>/dev/null)
  if [ -n "$current_head_sha_final" ] && [ "$current_head_sha_final" != "$HEAD_SHA" ]; then
    echo "::notice::PR #$PR's head moved from ${HEAD_SHA12:-unknown} to ${current_head_sha_final:0:12} immediately before this pause would have been attached -- standing down without pausing the replacement revision."
    return 0
  fi

  # GH_TOKEN="$READ_TOKEN" on the create call too: this step's default GH_TOKEN is
  # CODEX_REVIEW_PAT, not github.token -- and the scenario that lands us in a
  # fail-closed branch calling this helper in the first place can be EXACTLY
  # "CODEX_REVIEW_PAT is broken" (the merge call itself failed on that same
  # token). READ_TOKEN (github.token) already has issues: write per this
  # workflow's own top-level permissions block, which covers label creation.
  GH_TOKEN="$READ_TOKEN" gh label create needs-human --color d73a4a \
    --description "Needs a human decision -- an automated handler could not safely proceed" \
    --repo "$REPO" >/dev/null 2>&1 || true
  GH_TOKEN="$READ_TOKEN" gh api --method POST "repos/$REPO/issues/${PR}/labels" -f "labels[]=needs-human" >/dev/null 2>&1 || true
  local existing_labels
  existing_labels=$(GH_TOKEN="$READ_TOKEN" gh api --paginate "repos/$REPO/issues/${PR}/labels" --jq '.[].name')
  if grep -qxF "needs-human" <<< "$existing_labels"; then
    # Ownership check (Codex P1, round 1): a pre-POST "already_present" snapshot is racy --
    # if a maintainer applies needs-human manually in the gap between that snapshot and this
    # POST call, the snapshot still reads false, and this helper would then post its own
    # queue-owned SHA marker over the maintainer's pause. needs_human_is_stale() would later
    # see a marker postdating the maintainer's own labeling event with an older SHA and
    # auto-clear their manual pause as "stale" on the very next push. Fixed post-hoc instead
    # of pre-emptively: the label's own most recent `labeled` timeline event, fetched AFTER
    # our POST above, definitively shows who owns the CURRENT application -- our idempotent
    # POST creates no new timeline event when the label was already present, so this reflects
    # the true origin regardless of how the race landed. github-actions[bot] is what
    # GH_TOKEN=github.token (READ_TOKEN) always posts as.
    local raw owner_login
    if raw=$(GH_TOKEN="$READ_TOKEN" gh api --paginate --slurp "repos/$REPO/issues/${PR}/timeline" 2>/dev/null); then
      owner_login=$(printf '%s' "$raw" | jq -r '(add // []) | [.[] | select(.event == "labeled" and .label.name == "needs-human")] | if length > 0 then (last | .actor.login) else "" end' 2>/dev/null)
    else
      owner_login=""
    fi
    # Staleness marker: needs-human has no automatic clearing mechanism -- it's a
    # human decision point by design -- but without recording WHICH revision it
    # was attached for, a later push that fixes the problem stays silently
    # stalled: the label is PR-wide, not SHA-scoped. auto-merge-trigger.yml scans
    # for this marker to tell a genuinely-stale handoff (the head has since moved)
    # from one still covering the PR's current head. Best-effort (`|| true`): if
    # this fails to post, staleness detection just stays conservative (keeps
    # pausing), the same safe default as before this feature existed.
    #
    # Marker omitted whenever we're not the confirmed owner (including when the
    # ownership check itself couldn't be confirmed -- fail closed, same as every
    # other verification in this file) -- this call didn't establish the pause
    # and must not claim provenance over it.
    if [ "$owner_login" = "github-actions[bot]" ]; then
      GH_TOKEN="$READ_TOKEN" gh pr comment "$PR" --repo "$REPO" --body "<!-- queue-retry-handler:needs-human sha=$HEAD_SHA12 -->" >/dev/null 2>&1 || true
    else
      GH_TOKEN="$READ_TOKEN" gh pr comment "$PR" --repo "$REPO" --body "needs-human was already attached (owned by @${owner_login:-an unconfirmed source}, not this workflow) -- not claiming provenance over it; the existing pause stands as-is." >/dev/null 2>&1 || true
    fi
    return 0
  fi
  echo "::error::Could not attach needs-human to PR #$PR -- the handoff comment posted, but the label signal did not land."
  return 1
}

# The three `gh pr comment` calls below carry $NO_MORE_RETRIES_MARKER / a
# head-mismatch marker -- that's machine-readable retry STATE (check-retried's
# fallback consumes it), not merely a notification, so silently losing it on a
# transient comment failure would recreate the unbounded-retry bug the marker
# exists to prevent. Verify the marker actually landed and fail closed if it
# didn't; every OTHER `gh pr comment` call in this file stays best-effort
# (`|| true`), since only these three carry load-bearing state.
verify_marker_or_fail() {
  local marker="$1"
  local bodies
  bodies=$(GH_TOKEN="$READ_TOKEN" gh api --paginate "repos/$REPO/issues/${PR}/comments" --jq '.[] | select(.user.login == "github-actions[bot]") | .body')
  if [[ "$bodies" == *"$marker"* ]]; then
    return 0
  fi
  echo "::error::Could not confirm the no-more-retries marker was recorded for PR #$PR -- this is load-bearing retry state, not just a notification."
  return 1
}

# Consolidated (Codex P1, round 9, after this exact class of bug -- a code path that observes
# a live pause condition but exits without verifying disarm -- recurred independently across
# five separate rounds of this rollout's own review): every site in this script that needs to
# guarantee a PR is actually disarmed/dequeued now calls this ONE function instead of
# duplicating the sequence inline, so a future call site can't forget the fail-closed
# verification the way earlier rounds repeatedly did. Fails (returns 1) both when the PR is
# confirmed still armed/queued AND when either verification query itself can't be trusted (a
# `gh api` failure, or non-true/false JSON) -- treated identically as "cannot confirm safe."
disarm_dequeue_and_verify_or_fail() {
  local reason="$1"
  GH_TOKEN="$READ_TOKEN" gh pr merge "$PR" --repo "$REPO" --disable-auto >/dev/null 2>&1 || true
  local repo_owner="${REPO%%/*}" repo_name="${REPO##*/}"
  local queue_state pr_node_id in_queue still_armed
  if ! still_armed=$(GH_TOKEN="$READ_TOKEN" gh pr view "$PR" --repo "$REPO" --json autoMergeRequest --jq '.autoMergeRequest != null' 2>/dev/null); then
    still_armed=""
  fi
  if ! queue_state=$(GH_TOKEN="$READ_TOKEN" gh api graphql -f query='
    query($owner: String!, $name: String!, $pr: Int!) {
      repository(owner: $owner, name: $name) { pullRequest(number: $pr) { id isInMergeQueue } }
    }' -f owner="$repo_owner" -f name="$repo_name" -F pr="$PR" 2>/dev/null); then
    pr_node_id=""; in_queue=""
  else
    pr_node_id=$(printf '%s' "$queue_state" | jq -r '.data.repository.pullRequest.id' 2>/dev/null)
    in_queue=$(printf '%s' "$queue_state" | jq -r '.data.repository.pullRequest.isInMergeQueue' 2>/dev/null)
  fi
  if [ "$in_queue" = "true" ] && [ -n "$pr_node_id" ] && [ "$pr_node_id" != "null" ]; then
    echo "::notice::PR #$PR is in the native merge queue -- dequeuing."
    GH_TOKEN="$READ_TOKEN" gh api graphql -f query='mutation($id: ID!) { dequeuePullRequest(input: {id: $id}) { clientMutationId } }' -f id="$pr_node_id" >/dev/null 2>&1 || true
    local in_queue_state
    if ! in_queue_state=$(GH_TOKEN="$READ_TOKEN" gh api graphql -f query='
      query($owner: String!, $name: String!, $pr: Int!) {
        repository(owner: $owner, name: $name) { pullRequest(number: $pr) { isInMergeQueue } }
      }' -f owner="$repo_owner" -f name="$repo_name" -F pr="$PR" 2>/dev/null); then
      in_queue=""
    else
      in_queue=$(printf '%s' "$in_queue_state" | jq -r '.data.repository.pullRequest.isInMergeQueue' 2>/dev/null)
    fi
  fi
  if { [ "$still_armed" != "true" ] && [ "$still_armed" != "false" ]; } || { [ "$in_queue" != "true" ] && [ "$in_queue" != "false" ]; }; then
    echo "::error::Could not confirm PR #$PR's armed/queued state ($reason) -- treating as unsafe (fail closed)." >&2
    return 1
  fi
  if [ "$still_armed" = "true" ] || [ "$in_queue" = "true" ]; then
    echo "::error::Auto-merge is STILL ARMED and/or PR #$PR is still in the merge queue (armed: $still_armed, queued: $in_queue) despite the disable-auto/dequeue calls ($reason)." >&2
    return 1
  fi
  return 0
}

# Ordering matters: if the label were applied before this call and the call then
# failed, the revision would be permanently marked as having used its retry
# despite never actually being re-enqueued. Re-arm first; only mark the retry as
# spent once the merge call itself has actually succeeded.
#
# --match-head-commit: without it, a race between the pr-state fetch above and
# this call -- a contributor pushes a new commit in that window -- would silently
# arm auto-merge on the NEW head while the retry-used label still gets computed
# and applied against the OLD head_sha12. Pinning the mutation to the exact SHA
# fetched in pr-state makes it fail explicitly instead.
#
# Needs-human live check: pr-state's own fetch runs, then a 10s debounce, then
# dedup and check-retried each make their own API round-trips before this step
# even starts -- a maintainer can attach needs-human in that gap. Staleness-aware,
# not a bare presence check: auto-merge-trigger.yml clears needs-human when it
# determines it's stale for a new head, which normally keeps this retry from
# seeing a stale label at all -- but if that clear call itself failed transiently,
# a bare presence check here would wrongly refuse to retry a revision that was
# never actually paused. Only treat it as stale when a marker is provably tied to
# the label's own most recent application AND names an older sha than this
# retry's HEAD_SHA12; fail closed (still pause) whenever that can't be confirmed.
needs_human_is_stale() {
  local current_head_sha12="$1"
  local last_labeled_at marker_info marker_created_at recorded_sha12
  local raw
  # --slurp + flatten (`add`) before selecting the latest event, fetch and filter
  # as two explicit &&-chained steps rather than `--jq` passed to `gh api` itself:
  # this CLI version rejects `--slurp` combined with `--jq`/`--template` outright.
  # Deliberately not a bare `cmd1 | cmd2` pipe either: this step's default shell
  # has no `pipefail`, so a pipe's exit status reflects only the last command.
  if raw=$(GH_TOKEN="$READ_TOKEN" gh api --paginate --slurp "repos/$REPO/issues/${PR}/timeline" 2>/dev/null); then
    last_labeled_at=$(printf '%s' "$raw" | jq -r '(add // []) | [.[] | select(.event == "labeled" and .label.name == "needs-human")] | if length > 0 then (last | .created_at) else "" end' 2>/dev/null)
  else
    last_labeled_at=""
  fi
  [ -z "$last_labeled_at" ] && return 1
  if raw=$(GH_TOKEN="$READ_TOKEN" gh api --paginate --slurp "repos/$REPO/issues/${PR}/comments" 2>/dev/null); then
    marker_info=$(printf '%s' "$raw" | jq -r '(add // []) | [.[] | select(.user.login == "github-actions[bot]") | select(.body | test("queue-retry-handler:needs-human sha=[0-9a-f]{12}")) | [.created_at, (.body | capture("queue-retry-handler:needs-human sha=(?<s>[0-9a-f]{12})").s)] | @tsv] | if length > 0 then last else "" end' 2>/dev/null)
  else
    marker_info=""
  fi
  [ -z "$marker_info" ] && return 1
  marker_created_at=$(cut -f1 <<< "$marker_info")
  recorded_sha12=$(cut -f2 <<< "$marker_info")
  [ -z "$recorded_sha12" ] && return 1
  [[ "$marker_created_at" < "$last_labeled_at" ]] && return 1
  [ "$recorded_sha12" = "$current_head_sha12" ] && return 1
  return 0
}

current_labels=$(GH_TOKEN="$READ_TOKEN" gh api --paginate "repos/$REPO/issues/${PR}/labels" --jq '.[].name')
if grep -qxF "needs-human" <<< "$current_labels"; then
  if needs_human_is_stale "$HEAD_SHA12"; then
    echo "::notice::needs-human on PR #$PR is stale (provably tied to an earlier revision, not this retry's head $HEAD_SHA12) -- clearing it and proceeding with this head's retry."
    GH_TOKEN="$READ_TOKEN" gh api --method DELETE "repos/$REPO/issues/${PR}/labels/needs-human" >/dev/null 2>&1 || true
    labels_after_delete=$(GH_TOKEN="$READ_TOKEN" gh api --paginate "repos/$REPO/issues/${PR}/labels" --jq '.[].name')
    if grep -qxF "needs-human" <<< "$labels_after_delete"; then
      echo "::error::Could not remove the stale needs-human label from PR #$PR -- refusing to retry-merge while it remains attached. Remove it by hand, then rerun this job." >&2
      exit 1
    fi
  else
    # Codex P1, round 9: verify disarm before standing down, not just log -- a pending-run
    # eviction (GitHub retains only one pending run per concurrency group) can replace the
    # `labeled` event's own disarm run before it executes, so an admission run elsewhere could
    # have armed/queued the PR after its own label check but before this retry acquired the
    # shared lock. This run, observing the pause live, is positioned to finish that disarm.
    echo "::notice::PR #$PR now has needs-human covering its current head (or its provenance could not be confirmed) -- standing down without consuming this head's retry budget; ensuring auto-merge is not armed/queued despite the pause."
    disarm_dequeue_and_verify_or_fail "needs-human covering current head" || exit 1
    exit 0
  fi
fi

# Always CODEX_REVIEW_PAT: this call never passes --admin, so it never invokes bypass. This
# repo's main-protection ruleset already carries `thagale` as a permanent bypass_mode: always
# actor (unrelated to this cutover), but that bypass is a break-glass fallback for a stuck
# required check with no other recovery, not something this routine retry path reaches for.
# GITHUB_TOKEN would work equally for merge permissions, but a merge attributed to it
# suppresses the `push` event GitHub's anti-recursion guard would otherwise fire on main
# (auto-merge-trigger.yml's own header comment documents this exact failure mode) -- so
# CODEX_REVIEW_PAT avoids that suppression uniformly.
MERGE_TOKEN="$GH_TOKEN"

# Trust gate re-check (Codex P1, propagation migration): this script re-arms an
# ALREADY-admitted PR after a merge-group CI failure, but admission trust for a
# dependabot[bot]-authored PR is conditional on the `dependabot-vetted` label, which can
# be removed (reclassification to major) at any point between the original admission
# and this retry -- including during the failed merge-group attempt itself. Without
# re-checking here, a PR that lost its vetting mid-flight would get blindly re-armed by
# this retry, bypassing the same trust predicate auto-merge-trigger.yml enforces at
# every other admission point. thagale and catalyst-cloud-connector[bot] need no label
# check (matches auto-merge-trigger.yml's own is_trusted_author()).
pr_author=$(GH_TOKEN="$READ_TOKEN" gh pr view "$PR" --repo "$REPO" --json author --jq '.author.login')
pr_labels=$(GH_TOKEN="$READ_TOKEN" gh api --paginate "repos/${REPO}/issues/${PR}/labels" --jq '.[].name')
is_trusted_author() {
  local author="$1" current_labels="$2"
  if [ "$author" = "thagale" ] || [ "$author" = "catalyst-cloud-connector[bot]" ]; then
    return 0
  fi
  if [ "$author" = "dependabot[bot]" ]; then
    grep -qxF "dependabot-vetted" <<< "$current_labels" && return 0
  fi
  return 1
}

# Re-check the merge_queue ruleset rule AND the base immediately before mutating
# (Codex P2, round 5, extending the admission-gating fix already applied to both
# admission workflows' own arming calls): --match-head-commit pins the head
# atomically, but neither it nor the earlier pr-state fetch pins whether the
# merge_queue rule is still live -- if the rule were removed after pr-state ran (a
# cutover rollback), this call would enable ordinary direct auto-merge instead of
# queue admission, recreating the two-authority race with `.mergify.yml` while it's
# still active. `gh pr merge --auto` also takes no base argument at all, it just arms
# for the PR AS IT CURRENTLY EXISTS -- re-fetching both right here shrinks the window
# between these checks and the mutation call itself.
merge_queue_live=$(GH_TOKEN="$READ_TOKEN" gh api "repos/${REPO}/rules/branches/main" --jq '[.[] | select(.type == "merge_queue")] | length > 0' 2>/dev/null || echo false)
current_base=$(GH_TOKEN="$READ_TOKEN" gh pr view "$PR" --repo "$REPO" --json baseRefName --jq .baseRefName)
if [ "$merge_queue_live" != "true" ]; then
  echo "::notice::No merge_queue ruleset rule exists for main as of this retry attempt -- standing down without retrying or consuming this head's retry budget on PR #$PR."
elif [ "$current_base" != "main" ]; then
  echo "::notice::PR #$PR's base changed to '$current_base' (was main) since pr-state ran -- auto-merge-trigger.yml itself would refuse this base too. Standing down without consuming this head's retry budget."
elif ! is_trusted_author "$pr_author" "$pr_labels"; then
  echo "::notice::PR #$PR's author ($pr_author) is no longer trusted for auto-admission as of this retry attempt (e.g. dependabot-vetted was removed after a reclassification to major) -- disarming/dequeuing instead of retrying, and not consuming this head's retry budget."
  disarm_dequeue_and_verify_or_fail "author no longer trusted at retry time" || exit 1
elif GH_TOKEN="$MERGE_TOKEN" gh pr merge "$PR" --repo "$REPO" --auto --squash --match-head-commit "$HEAD_SHA"; then
  # GH_TOKEN="$READ_TOKEN" here too: a successful merge call only proves
  # CODEX_REVIEW_PAT has pull-requests:write -- it says nothing about whether it
  # ALSO has issues:write, a genuinely separate scope for a fine-grained PAT.
  GH_TOKEN="$READ_TOKEN" gh label create "$LABEL" --color d4c5f9 \
    --description "This revision already used its one automatic queue retry" \
    --repo "$REPO" >/dev/null 2>&1 || true
  GH_TOKEN="$READ_TOKEN" gh api --method POST "repos/$REPO/issues/${PR}/labels" -f "labels[]=$LABEL" >/dev/null 2>&1 || true

  existing_labels=$(GH_TOKEN="$READ_TOKEN" gh api --paginate "repos/$REPO/issues/${PR}/labels" --jq '.[].name')
  attached=$(grep -qxF "$LABEL" <<< "$existing_labels" && echo true || echo false)
  if [ "$attached" != "true" ]; then
    current_head_sha=$(GH_TOKEN="$READ_TOKEN" gh pr view "$PR" --repo "$REPO" --json headRefOid --jq .headRefOid)
    if [ "$current_head_sha" != "$HEAD_SHA" ]; then
      echo "::notice::PR #$PR's head moved to ${current_head_sha:0:12} while verifying the retry-used label for $HEAD_SHA12 -- that commit's own retry budget is unaffected; standing down without touching auto-merge on the replacement revision."
      exit 0
    fi
    echo "::error::Retry-used label failed to attach for PR #$PR (commit $HEAD_SHA12) -- disarming auto-merge and handing off instead of risking an unbounded retry loop."
    NO_MORE_RETRIES_MARKER="<!-- queue-retry-handler:no-more-retries sha=$HEAD_SHA12 -->"
    # Fail the step outright on a disarm failure rather than falling through to the marker/
    # label calls below -- if THOSE happen to succeed, the step (and this whole fail-closed
    # handoff) would report success while the PR is still armed or queued, or its state was
    # never confirmed. The native queue has no label condition of its own, so an armed PR can
    # merge before a human ever sees the warning comment.
    if ! disarm_dequeue_and_verify_or_fail "retry-used label failed to attach"; then
      GH_TOKEN="$READ_TOKEN" gh pr comment "$PR" --repo "$REPO" --body "Merge-group CI failed for this revision (commit \`$HEAD_SHA12\`), and the retry-tracking label could not be recorded. Attempted to disarm auto-merge, but could NOT confirm it's actually clear (or the verification itself failed) -- this needs urgent human attention: run \`gh pr merge --disable-auto\` yourself and verify, or close/reopen the PR. $NO_MORE_RETRIES_MARKER" || true
      exit 1
    fi
    GH_TOKEN="$READ_TOKEN" gh pr comment "$PR" --repo "$REPO" --body "Merge-group CI failed for this revision (commit \`$HEAD_SHA12\`), and the retry-tracking label could not be recorded. Disarmed auto-merge rather than risk retrying unboundedly -- this needs a human: push a fix, or manually run \`gh pr merge --auto --squash\` once the labeling problem is resolved. $NO_MORE_RETRIES_MARKER" || true
    verify_marker_or_fail "$NO_MORE_RETRIES_MARKER" || exit 1
    attach_needs_human_or_fail || exit 1
  else
    GH_TOKEN="$READ_TOKEN" gh pr comment "$PR" --repo "$REPO" --body "Merge-group CI failed for this revision (commit \`$HEAD_SHA12\`) -- automatically re-armed auto-merge for one retry. If this fails again for the same commit, it will NOT retry again; it needs a human to look." || true
  fi
else
  current_head=$(GH_TOKEN="$READ_TOKEN" gh pr view "$PR" --repo "$REPO" --json headRefOid --jq .headRefOid)
  if [ "$current_head" != "$HEAD_SHA" ]; then
    echo "::notice::PR #$PR's head moved from $HEAD_SHA12 to ${current_head:0:12} during this run -- the retry for the OLD commit is moot; the new commit has its own retry budget. No action needed."
  else
    # Codex P1, round 7: disarm/dequeue-and-verify BEFORE completing this handoff,
    # same as the retry-used-label-failed branch above -- a `gh pr merge --auto` call
    # can reach GitHub and actually arm/queue the PR even though the CLI reports
    # nonzero (a lost response, not a rejected request; `gh pr merge --help` documents
    # that --auto either enables auto-merge or, once checks have passed, adds the PR
    # straight to the merge queue). Without this, that path could complete
    # successfully and mark the attempt handled while the PR remains armed or queued
    # despite carrying needs-human -- and the label POST below uses github.token,
    # whose resulting `labeled` event can't self-trigger a disarm run anyway (GitHub's
    # own GITHUB_TOKEN anti-recursion guard), so nothing else would catch this either.
    echo "::error::Could not confirm auto-merge re-armed for PR #$PR (head unchanged at $HEAD_SHA12) -- disarming/dequeuing (the merge call may have partially succeeded despite the nonzero exit) and handing off instead of silently stalling."
    CURRENT_HEAD_MARKER="<!-- queue-retry-handler:no-more-retries sha=$HEAD_SHA12 -->"
    if ! disarm_dequeue_and_verify_or_fail "retry's re-arm call reported an error"; then
      GH_TOKEN="$READ_TOKEN" gh pr comment "$PR" --repo "$REPO" --body "Merge-group CI failed for commit \`$HEAD_SHA12\`, and the automatic retry's own re-arm call reported an error. Attempted to disarm/dequeue, but could NOT confirm it's actually clear (or the verification itself failed) -- this needs urgent human attention: run \`gh pr merge --disable-auto\` yourself and verify (and check whether it's still in the merge queue), or close/reopen the PR. $CURRENT_HEAD_MARKER" || true
      exit 1
    fi
    GH_TOKEN="$READ_TOKEN" gh pr comment "$PR" --repo "$REPO" --body "Merge-group CI failed for commit \`$HEAD_SHA12\`, and the automatic retry itself could not re-arm auto-merge (API error). This needs a human: push a fix, or manually run \`gh pr merge --auto --squash\`. $CURRENT_HEAD_MARKER" || true
    verify_marker_or_fail "$CURRENT_HEAD_MARKER" || exit 1
    attach_needs_human_or_fail || exit 1
  fi
fi
