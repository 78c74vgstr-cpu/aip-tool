import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter
import pandas as pd
import json
import re
import io
import time
import concurrent.futures
from threading import Lock
from openai import OpenAI

# ── Styles ─────────────────────────────────────────────────────────────────────
YELLOW_FILL   = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")  # cell updated
RED_FILL      = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")  # unmappable SOC note
GREEN_FILL    = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")  # new task row
GREY_FILL     = PatternFill(start_color="BFBFBF", end_color="BFBFBF", fill_type="solid")  # deleted task row
BLUE_FILL     = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")  # header
WHITE_FONT    = Font(color="FFFFFF", bold=True)
WRAP_ALIGN    = Alignment(wrap_text=True, vertical="top")

# ── Column name mapping: AMP column → MPD column ───────────────────────────────
# The engine reads new values FROM the MPD and writes them INTO the AMP.
# AMP and MPD use different names for the same columns — this table bridges them.
AMP_TO_MPD_COL = {
    "INTERVAL":              "SAMPLE\nINTERVAL",
    "100% INTERVAL":         "100%\nINTERVAL",
    "100% THRESHOLD":        "100%\nTHRESHOLD",
    "DESCRIPTION":           "DESCRIPTION",
    "ACCESS":                "ACCESS",
    "PREPARATION":           "PREPARATION",
    "ZONE":                  "ZONE",
    "TASK CODE":             "TASK CODE",
    "SOURCE TASK\nREFERENCE": "SOURCE TASK\nREFERENCE",
    "REFERENCE":             "REFERENCE",
}

# ── SOC text field → AMP column ────────────────────────────────────────────────
# When we parse "FIELD CHANGED FROM X TO Y" out of the SOC description,
# this maps those field labels to the AMP column we should update.
SOC_FIELD_TO_AMP_COL = {
    "100% INTERVAL":  "100% INTERVAL",
    "100% THRESHOLD": "100% THRESHOLD",
    "INTERVAL":       "INTERVAL",
    "THRESHOLD":      "100% THRESHOLD",
    "TASK CODE":      "TASK CODE",
    "ZONE":           "ZONE",
    "ACCESS":         "ACCESS",
    "PREPARATION":    "PREPARATION",
    "REFERENCE":      "REFERENCE",
    "DESCRIPTION":    "DESCRIPTION",
    "TASK TITLE":     "DESCRIPTION",
}

# Fields in SOC that have no matching AMP column — always flag for manual review
UNMAPPABLE_FIELDS = {
    "APPLICABILITY", "TASK NOTE", "TASK MH", "ACCESS MH", "PREP MH",
    "IN PREPARATION MH", "TPS MARKER", "SKILL", "MEN",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def find_header_row(ws, max_search=20):
    best_row, best_count = 1, 0
    for row in ws.iter_rows(min_row=1, max_row=max_search):
        count = sum(1 for cell in row if cell.value not in (None, ""))
        if count > best_count:
            best_count = count
            best_row   = row[0].row
    headers = {}
    for cell in ws[best_row]:
        if cell.value not in (None, ""):
            headers[str(cell.value).strip()] = cell.column
    return best_row, headers


def build_task_index(ws, header_row, task_col_idx):
    index = {}
    for col in ws.iter_cols(min_col=task_col_idx, max_col=task_col_idx, min_row=header_row + 1):
        for cell in col:
            if cell.value not in (None, ""):
                index[str(cell.value).strip()] = cell.row
    return index


def read_mpd_dataframe(file_bytes):
    raw        = pd.read_excel(file_bytes, header=None, nrows=25)
    header_idx = int(raw.notna().sum(axis=1).idxmax())
    file_bytes.seek(0)
    df         = pd.read_excel(file_bytes, header=header_idx)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_text(val):
    """Collapse all whitespace variations for comparison."""
    return re.sub(r'\s+', ' ', str(val or "")).strip()


def parse_soc_field_changes(description: str):
    """
    Parse 'FIELD CHANGED FROM: "X", TO: "Y"' patterns from SOC description text.
    Returns list of (field_upper, old_value, new_value) tuples.
    """
    results = []
    clean   = re.sub(r'\n+', '\n', str(description))
    pattern = r'^([\w][^\n]{0,50}?)\s+CHANGED FROM:\s*"?(.*?)"?,?\s*TO:\s*"?(.*?)"?\s*$'
    for m in re.finditer(pattern, clean, re.IGNORECASE | re.MULTILINE):
        field   = m.group(1).strip().upper()
        old_val = m.group(2).strip().strip('"').strip("'")
        new_val = m.group(3).strip().strip('"').strip("'")
        if new_val:
            results.append((field, old_val, new_val))
    return results


def classify_soc_changes(description: str):
    """
    Given a SOC description, return:
      - updates:     {amp_col: new_value}  — changes we can apply automatically
      - unmappable:  [str]                 — changes that need manual review
    """
    updates    = {}
    unmappable = []

    for field, old_val, new_val in parse_soc_field_changes(description):
        # Check exact match in SOC→AMP map
        if field in SOC_FIELD_TO_AMP_COL:
            amp_col = SOC_FIELD_TO_AMP_COL[field]
            updates[amp_col] = new_val

        # Check if it's a known unmappable field
        elif any(uf in field for uf in UNMAPPABLE_FIELDS):
            unmappable.append(f"{field}: \"{old_val}\" → \"{new_val}\"")

        # Zone-specific MH patterns like "IN ZONE \"147\" TASK MH"
        elif re.search(r'IN ZONE .* (TASK MH|ACCESS MH|PREP MH|MEN)', field):
            unmappable.append(f"{field}: \"{old_val}\" → \"{new_val}\"")

        else:
            # Unknown field — flag for manual review
            unmappable.append(f"{field}: \"{old_val}\" → \"{new_val}\"")

    # Also catch ADDED/REMOVED patterns that don't fit CHANGED FROM/TO
    added_pattern   = r'([\w][^\n]{0,40}?)\s+["\'](.+?)["\']\s+ADDED'
    removed_pattern = r'([\w][^\n]{0,40}?)\s+["\'](.+?)["\']\s+REMOVED'
    for m in re.finditer(added_pattern, str(description), re.IGNORECASE):
        field = m.group(1).strip().upper()
        val   = m.group(2).strip()
        if any(uf in field for uf in UNMAPPABLE_FIELDS) or 'TASK' in field or 'APPLICABILITY' in field:
            unmappable.append(f"{field} ADDED: \"{val}\"")
    for m in re.finditer(removed_pattern, str(description), re.IGNORECASE):
        field = m.group(1).strip().upper()
        val   = m.group(2).strip()
        if any(uf in field for uf in UNMAPPABLE_FIELDS) or 'TASK' in field or 'APPLICABILITY' in field:
            unmappable.append(f"{field} REMOVED: \"{val}\"")

    # Applicability changes (not caught by CHANGED FROM pattern sometimes)
    if 'APPLICABILITY' in str(description).upper() and 'APPLICABILITY' not in [f for f, _, _ in parse_soc_field_changes(description)]:
        for m in re.finditer(r'APPLICABILITY\s+CHANGED\s+FROM:\s*"?(.*?)"?,?\s*TO:\s*"?(.*?)"?\s*$',
                             str(description), re.IGNORECASE | re.MULTILINE):
            unmappable.append(f"APPLICABILITY: \"{m.group(1).strip()}\" → \"{m.group(2).strip()}\"")

    # Deduplicate
    unmappable = list(dict.fromkeys(unmappable))
    return updates, unmappable


# ── LLM Parsing (for N/D detection and fallback) ──────────────────────────────

_BATCH_SIZE   = 20
_CALL_TIMEOUT = 90
_MAX_RETRIES  = 3
_MAX_WORKERS  = 5


def _call_llm_with_retry(client, system_prompt, numbered, batch_num, timeout, max_retries):
    for attempt in range(1, max_retries + 1):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(
                    client.chat.completions.create,
                    model           = "gpt-4o-mini",
                    messages        = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": f"Process these SOC rows:\n\n{numbered}"},
                    ],
                    response_format = {"type": "json_object"},
                    temperature     = 0,
                )
                resp = future.result(timeout=timeout)
            parsed = json.loads(resp.choices[0].message.content)
            return parsed.get("tasks", parsed if isinstance(parsed, list) else [])
        except concurrent.futures.TimeoutError:
            wait = 2 ** attempt
            print(f"[LLM] Batch {batch_num} timeout (attempt {attempt}), retrying in {wait}s")
            time.sleep(wait)
        except Exception as exc:
            wait = 2 ** attempt
            print(f"[LLM] Batch {batch_num} error (attempt {attempt}): {exc}, retrying in {wait}s")
            time.sleep(wait)
    print(f"[LLM] Batch {batch_num} permanently failed — skipping.")
    return []


def parse_soc_with_llm(soc_file_bytes, aip_columns, api_key,
                        progress_callback=None, checkpoint_state=None):
    """
    Use GPT only for what regex can't do reliably: detecting new/deleted tasks
    and extracting the raw SOC note per task.

    The actual column updates and unmappable flagging are done by classify_soc_changes()
    using direct regex parsing — faster, cheaper, and more reliable.
    """
    client = OpenAI(api_key=api_key)

    soc_df         = pd.read_excel(soc_file_bytes)
    soc_df.columns = [str(c).strip() for c in soc_df.columns]

    rows_text = []
    for _, row in soc_df.iterrows():
        parts = [f"{col}: {val}" for col, val in row.items()
                 if pd.notna(val) and str(val).strip()]
        rows_text.append(" | ".join(parts))

    system_prompt = f"""You are an expert in aircraft maintenance documentation.

For each SOC row return a JSON entry with ONLY these fields:
- "task_id"        : Task ID from "TASK NUMBER" or similar field
- "mvt"            : The MVT/movement code exactly as given (R, N, or D)
- "soc_note"       : Complete raw description text from this SOC row, verbatim
- "is_new_task"    : true if MVT is "N" (new task)
- "is_deleted_task": true if MVT is "D" (deleted/removed task)

Return ONLY: {{"tasks": [...]}}
Skip rows with no Task ID.
Do NOT attempt to identify which columns changed — that is handled separately.
"""

    batches = []
    for i in range(0, len(rows_text), _BATCH_SIZE):
        chunk    = rows_text[i: i + _BATCH_SIZE]
        numbered = "\n".join(f"{j+1}. {r}" for j, r in enumerate(chunk))
        batches.append((i // _BATCH_SIZE + 1, numbered))

    total_batches = len(batches)
    completed     = {}

    if checkpoint_state is not None:
        completed = checkpoint_state.get("soc_parse_checkpoint", {})
        if completed:
            print(f"[Checkpoint] Resuming — {len(completed)}/{total_batches} batches done.")

    remaining = [(num, text) for num, text in batches if num not in completed]
    lock       = Lock()

    def process_batch(args):
        batch_num, numbered = args
        tasks = _call_llm_with_retry(client, system_prompt, numbered,
                                     batch_num, _CALL_TIMEOUT, _MAX_RETRIES)
        with lock:
            completed[batch_num] = tasks
            if checkpoint_state is not None:
                checkpoint_state["soc_parse_checkpoint"] = completed.copy()
        return batch_num, tasks

    with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = [executor.submit(process_batch, args) for args in remaining]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            if progress_callback:
                progress_callback(len(completed), total_batches)

    all_llm_results = []
    for batch_num in sorted(completed.keys()):
        all_llm_results.extend(completed[batch_num])

    if checkpoint_state is not None:
        checkpoint_state.pop("soc_parse_checkpoint", None)

    # Now enrich each LLM result with regex-parsed column updates
    # Build a lookup from task_id → SOC row for fast access
    soc_df["TASK NUMBER"] = soc_df.get("TASK NUMBER", soc_df.iloc[:, 1]).astype(str).str.strip()
    soc_lookup = soc_df.set_index("TASK NUMBER").to_dict(orient="index")

    enriched = []
    for entry in all_llm_results:
        task_id = str(entry.get("task_id", "")).strip()
        if not task_id:
            continue

        soc_row  = soc_lookup.get(task_id, {})
        raw_desc = str(soc_row.get("DESCRIPTION", entry.get("soc_note", "")))

        updates, unmappable = classify_soc_changes(raw_desc)

        enriched.append({
            "task_id":          task_id,
            "mvt":              entry.get("mvt", "R"),
            "soc_note":         raw_desc,
            "is_new_task":      entry.get("is_new_task", False),
            "is_deleted_task":  entry.get("is_deleted_task", False),
            "column_updates":   updates,      # {amp_col: new_value}
            "unmappable":       unmappable,   # [str] for manual review
        })

    return enriched


# ── AIP Updater ────────────────────────────────────────────────────────────────

def update_aip(aip_file_bytes, mpd_df: pd.DataFrame,
               task_changes: list, task_id_col_name: str):
    """
    Update AIP from parsed SOC changes.

    Per-row behavior:
    - REV column added: shows R / N / D for easy filtering
    - SOC Note column added: shows ONLY unmappable change sentences (red if any, else blank)
    - Changed cells: YELLOW highlight, value replaced from MPD or SOC-parsed value
    - New task rows: GREEN full-row highlight
    - Deleted task rows: GREY full-row highlight
    - No changes: nothing added (clean rows stay clean)

    Returns: (workbook, change_log, flagged, summary_stats)
    """
    wb = openpyxl.load_workbook(aip_file_bytes)
    ws = wb.active

    header_row, aip_headers = find_header_row(ws)

    if task_id_col_name not in aip_headers:
        raise ValueError(
            f"Column '{task_id_col_name}' not found in AIP.\n"
            f"Found: {list(aip_headers.keys())}"
        )

    task_col_idx   = aip_headers[task_id_col_name]
    aip_task_index = build_task_index(ws, header_row, task_col_idx)

    # MPD lookup
    mpd_task_col = None
    for col in mpd_df.columns:
        if col.strip().lower() == task_id_col_name.strip().lower():
            mpd_task_col = col
            break
    if mpd_task_col is None:
        for col in mpd_df.columns:
            if "task" in col.lower():
                mpd_task_col = col
                break
    if mpd_task_col is None:
        raise ValueError(f"Task ID column not found in MPD.\nMPD columns: {list(mpd_df.columns)}")

    mpd_df               = mpd_df.copy()
    mpd_df[mpd_task_col] = mpd_df[mpd_task_col].astype(str).str.strip()
    mpd_lookup           = mpd_df.set_index(mpd_task_col).to_dict(orient="index")

    # ── Add REV and SOC Note columns to the right ────────────────────────────
    max_col      = ws.max_column
    rev_col      = max_col + 1
    soc_note_col = max_col + 2

    for col_idx, label in [(rev_col, "REV"), (soc_note_col, "SOC Note")]:
        cell      = ws.cell(row=header_row, column=col_idx, value=label)
        cell.fill = BLUE_FILL
        cell.font = WHITE_FONT

    ws.column_dimensions[get_column_letter(rev_col)].width      = 6
    ws.column_dimensions[get_column_letter(soc_note_col)].width = 55

    # ── Process each SOC entry ───────────────────────────────────────────────
    change_log = []
    flagged    = []
    stats      = {"updated": 0, "new": 0, "deleted": 0, "not_in_amp": 0, "manual_review": 0}

    soc_task_ids = {str(e.get("task_id", "")).strip() for e in task_changes if e.get("task_id")}

    for entry in task_changes:
        task_id        = str(entry.get("task_id", "")).strip()
        mvt            = str(entry.get("mvt", "R")).strip().upper()
        is_new         = entry.get("is_new_task", False)
        is_deleted     = entry.get("is_deleted_task", False)
        column_updates = entry.get("column_updates", {})   # {amp_col: new_val}
        unmappable     = entry.get("unmappable", [])
        soc_note_text  = entry.get("soc_note", "")

        if not task_id:
            continue

        # ── Task not in AMP ───────────────────────────────────────────────────
        if task_id not in aip_task_index:
            stats["not_in_amp"] += 1
            if is_new:
                flagged.append({"Task ID": task_id, "MVT": "N", "Issue": "New task — not yet in AMP, add manually"})
            elif not is_deleted:
                flagged.append({"Task ID": task_id, "MVT": mvt, "Issue": "Task in SOC but not found in AMP"})
            continue

        aip_row = aip_task_index[task_id]

        # ── Write REV code ────────────────────────────────────────────────────
        rev_cell       = ws.cell(row=aip_row, column=rev_col, value=mvt)
        rev_cell.alignment = Alignment(horizontal="center", vertical="center")

        # ── NEW TASK ──────────────────────────────────────────────────────────
        if is_new:
            for c in range(1, soc_note_col + 1):
                ws.cell(row=aip_row, column=c).fill = GREEN_FILL
            ws.cell(row=aip_row, column=soc_note_col).value     = "NEW TASK — review applicability"
            ws.cell(row=aip_row, column=soc_note_col).alignment = WRAP_ALIGN
            stats["new"] += 1
            continue

        # ── DELETED TASK ──────────────────────────────────────────────────────
        if is_deleted:
            for c in range(1, soc_note_col + 1):
                ws.cell(row=aip_row, column=c).fill = GREY_FILL
            ws.cell(row=aip_row, column=soc_note_col).value     = "DELETED IN SOC — review for removal"
            ws.cell(row=aip_row, column=soc_note_col).alignment = WRAP_ALIGN
            stats["deleted"] += 1
            continue

        # ── REVISED TASK ──────────────────────────────────────────────────────
        task_changed = False

        for amp_col, soc_new_val in column_updates.items():
            if amp_col not in aip_headers:
                continue

            aip_col_idx = aip_headers[amp_col]
            cell        = ws.cell(row=aip_row, column=aip_col_idx)
            old_val     = cell.value

            # Try to get the value from MPD first (most authoritative)
            # Fall back to the SOC-parsed value if MPD doesn't have it
            mpd_col  = AMP_TO_MPD_COL.get(amp_col)
            new_val  = soc_new_val  # default to SOC-parsed

            if task_id in mpd_lookup and mpd_col and mpd_col in mpd_lookup[task_id]:
                mpd_val = mpd_lookup[task_id][mpd_col]
                if mpd_val is not None and not (isinstance(mpd_val, float) and pd.isna(mpd_val)):
                    new_val = mpd_val

            # Normalize for comparison — skip if only whitespace differs
            old_norm = normalize_text(old_val)
            new_norm = normalize_text(new_val)

            if old_norm == new_norm:
                continue

            cell.value = new_val
            cell.fill  = YELLOW_FILL
            task_changed = True

            change_log.append({
                "Task ID":   task_id,
                "Column":    amp_col,
                "Old Value": old_norm,
                "New Value": new_norm,
            })

        if task_changed:
            stats["updated"] += 1

        # ── SOC Note: only write if there are unmappable changes ──────────────
        if unmappable:
            note_cell           = ws.cell(row=aip_row, column=soc_note_col)
            note_cell.value     = "\n".join(f"• {u}" for u in unmappable)
            note_cell.fill      = RED_FILL
            note_cell.alignment = WRAP_ALIGN
            stats["manual_review"] += 1

    # ── Change Log sheet ──────────────────────────────────────────────────────
    if "Change Log" in wb.sheetnames:
        del wb["Change Log"]
    log_ws = wb.create_sheet("Change Log")

    log_headers = ["Task ID", "Column", "Old Value", "New Value"]
    for c_idx, h in enumerate(log_headers, 1):
        cell      = log_ws.cell(row=1, column=c_idx, value=h)
        cell.fill = BLUE_FILL
        cell.font = WHITE_FONT

    for r_idx, e in enumerate(change_log, 2):
        log_ws.cell(row=r_idx, column=1, value=e["Task ID"])
        log_ws.cell(row=r_idx, column=2, value=e["Column"])
        log_ws.cell(row=r_idx, column=3, value=e["Old Value"])
        log_ws.cell(row=r_idx, column=4, value=e["New Value"])

    for col in log_ws.columns:
        width = max((len(str(cell.value or "")) for cell in col), default=10) + 4
        log_ws.column_dimensions[get_column_letter(col[0].column)].width = min(width, 60)

    # ── Legend sheet ──────────────────────────────────────────────────────────
    if "Legend" in wb.sheetnames:
        del wb["Legend"]
    leg_ws = wb.create_sheet("Legend")
    leg_ws.column_dimensions["A"].width = 20
    leg_ws.column_dimensions["B"].width = 60

    title      = leg_ws.cell(row=1, column=1, value="Colour Legend")
    title.font = Font(bold=True, size=13)

    legend_rows = [
        (YELLOW_FILL, "Yellow cell",  "Cell value was updated automatically from MPD"),
        (RED_FILL,    "Red SOC Note", "Change exists in SOC but has no AMP column — review manually"),
        (GREEN_FILL,  "Green row",    "New task added in this revision — review applicability"),
        (GREY_FILL,   "Grey row",     "Task deleted in this revision — review for removal"),
    ]
    for i, (fill, label, desc) in enumerate(legend_rows, start=3):
        a      = leg_ws.cell(row=i, column=1, value=label)
        a.fill = fill
        b      = leg_ws.cell(row=i, column=2, value=desc)
        b.fill = fill

    # ── Summary sheet ─────────────────────────────────────────────────────────
    if "Summary" in wb.sheetnames:
        del wb["Summary"]
    sum_ws = wb.create_sheet("Summary")
    sum_ws.column_dimensions["A"].width = 35
    sum_ws.column_dimensions["B"].width = 15

    sum_title      = sum_ws.cell(row=1, column=1, value="Revision Summary")
    sum_title.font = Font(bold=True, size=13)

    summary_rows = [
        ("Tasks in SOC",                          len(soc_task_ids)),
        ("Tasks updated automatically",           stats["updated"]),
        ("Cells updated (total)",                 len(change_log)),
        ("New tasks (green rows)",                stats["new"]),
        ("Deleted tasks (grey rows)",             stats["deleted"]),
        ("Tasks flagged for manual review (red)", stats["manual_review"]),
        ("Tasks in SOC not found in AMP",         stats["not_in_amp"]),
    ]
    for i, (label, val) in enumerate(summary_rows, start=3):
        sum_ws.cell(row=i, column=1, value=label)
        sum_ws.cell(row=i, column=2, value=val)

    return wb, change_log, flagged, stats
