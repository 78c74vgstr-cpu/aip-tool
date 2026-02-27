import streamlit as st
import io
import pandas as pd
import openpyxl

from engine import (
    find_header_row,
    parse_soc_with_llm,
    read_mpd_dataframe,
    update_aip,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AIP Revision Tool",
    page_icon  = "✈️",
    layout     = "centered"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type = "password",
        help = "Get yours at platform.openai.com → API keys"
    )
    task_id_col = st.text_input(
        "Task ID Column Name",
        value = "Task No",
        help  = "The exact column header used for Task IDs in your AIP and MPD"
    )

    st.divider()
    st.caption("💡 Cost: ~$0.10–$0.20 per 1,000 tasks")
    st.caption("🤖 Model: gpt-4o-mini")
    st.caption("📄 AIP must be .xlsx format")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("✈️  AIP Revision Tool")
st.caption(
    "Upload your SOC, MPD, and AIP — the tool will surgically apply all updates, "
    "highlight every changed cell, and generate a full change log."
)
st.divider()

# ── File uploads ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📄 SOC**")
    st.caption("Summary of Changes")
    soc_file = st.file_uploader("soc", type=["xlsx", "xls"], label_visibility="collapsed")
    if soc_file:
        st.success("✓ Ready")

with col2:
    st.markdown("**📄 MPD**")
    st.caption("Manufacturer's Planning Document")
    mpd_file = st.file_uploader("mpd", type=["xlsx", "xls"], label_visibility="collapsed")
    if mpd_file:
        st.success("✓ Ready")

with col3:
    st.markdown("**📄 AIP**")
    st.caption("Your Aircraft Inspection Program")
    aip_file = st.file_uploader("aip", type=["xlsx"], label_visibility="collapsed")
    if aip_file:
        st.success("✓ Ready")

st.divider()

# ── Run button ────────────────────────────────────────────────────────────────
all_ready = all([soc_file, mpd_file, aip_file, api_key])

if not all_ready:
    missing = []
    if not api_key:  missing.append("OpenAI API key (sidebar)")
    if not soc_file: missing.append("SOC file")
    if not mpd_file: missing.append("MPD file")
    if not aip_file: missing.append("AIP file (.xlsx)")
    st.info("Still needed: " + " · ".join(missing))

if st.button("▶  Run Update", type="primary", use_container_width=True, disabled=not all_ready):

    progress_bar = st.progress(0)
    status_text  = st.empty()

    try:
        # ── Step 1: Read AIP column names ────────────────────────────────────
        status_text.text("📖  Reading AIP structure…")
        aip_bytes      = io.BytesIO(aip_file.read())
        wb_temp        = openpyxl.load_workbook(aip_bytes)
        _, aip_headers = find_header_row(wb_temp.active)
        aip_col_names  = list(aip_headers.keys())
        aip_bytes.seek(0)
        progress_bar.progress(10)

        # ── Step 2: Parse SOC with LLM ───────────────────────────────────────
        status_text.text("🤖  Parsing SOC with AI…  (30–90 s for large files)")
        soc_bytes = io.BytesIO(soc_file.read())

        def llm_progress(batch_num, total):
            pct = 10 + int((batch_num / total) * 40)
            progress_bar.progress(pct)
            status_text.text(f"🤖  Parsing SOC — batch {batch_num} / {total}…")

        task_changes = parse_soc_with_llm(soc_bytes, aip_col_names, api_key, llm_progress)
        progress_bar.progress(50)

        # ── Step 3: Read MPD ─────────────────────────────────────────────────
        status_text.text("📖  Reading MPD…")
        mpd_bytes = io.BytesIO(mpd_file.read())
        mpd_df    = read_mpd_dataframe(mpd_bytes)
        progress_bar.progress(65)

        # ── Step 4: Update AIP ───────────────────────────────────────────────
        status_text.text("✏️  Applying updates to AIP…")
        updated_wb, change_log, flagged, stats = update_aip(
            aip_bytes, mpd_df, task_changes, task_id_col
        )
        progress_bar.progress(90)

        # ── Step 5: Save ─────────────────────────────────────────────────────
        status_text.text("💾  Saving…")
        output = io.BytesIO()
        updated_wb.save(output)
        output.seek(0)
        progress_bar.progress(100)
        status_text.empty()

        # ── Results ───────────────────────────────────────────────────────────
        st.success("✅  Update complete!")

        m1, m2, m3 = st.columns(3)
        m1.metric("Tasks Updated", len({e["Task ID"] for e in change_log}))
        m2.metric("Cells Changed", len(change_log))
        m3.metric("Flagged Items", len(flagged))

        st.download_button(
            label               = "⬇  Download Updated AIP",
            data                = output,
            file_name           = f"Updated_{aip_file.name}",
            mime                = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type                = "primary",
            use_container_width = True,
        )

        if flagged:
            with st.expander(f"⚠️  {len(flagged)} items flagged for manual review"):
                st.dataframe(pd.DataFrame(flagged), use_container_width=True)

        if change_log:
            with st.expander(f"📋  Change log preview  ({len(change_log)} changes)"):
                st.dataframe(pd.DataFrame(change_log), use_container_width=True)

    except Exception as exc:
        st.error(f"Something went wrong: {str(exc)}")
        st.exception(exc)
